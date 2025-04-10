import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import DataLoader
from   torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
import sys, os
import time
import numpy as np
import matplotlib.pylab as plt

LOG = False

def um2f(um):
    return 2.997924580e14/um

def f2um(freq):
    return 2.997924580e14/freq


# NET = [ 15, 15 ]
# NET = [ 13, 17, 13 ]
# NET = [ 15, 19, 15 ]
# NET = [ 17, 21, 17 ]

# MAXITER = 20000
# MAXITER = 12000
# MAXITER = 3000


def NN_fit(prefix, dustname, cells, nnabs, nnemit, file_absorbed, file_emitted, nngpu=1, nnmaxiter=3000, nnnet=[13, 17, 13]):
    """
    Given absorptions [cells, nnabs] and corresponding emissions [cells, nnemit],
    make a neural network fit and save weights to <prefix>_<dustnam>.nn
    Note: 
        - in A2E_MABU.py, file_absorbed has been divided by FF, to make NN
          mapping independent of the frequency grid (different when NN is fitted
          and when it is used to predict emission)
        - all data on absorptions and emissions is read to Python arrays,
          GPU memory limitations should be overcome by choosing a small enough
          training set
    """
    Na = len(nnabs)
    Ne = len(nnemit)
    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            if (len(nnnet)==2):
                self.layers = nn.Sequential(
                nn.Linear(Na, nnnet[0]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[0], nnnet[1]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[1], Ne)
                )
            elif (len(nnnet)==3):
                self.layers = nn.Sequential(
                nn.Linear(Na, nnnet[0]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[0], nnnet[1]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[1], nnnet[2]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[2], Ne)
                )                
            elif (len(nnnet)==4):
                self.layers = nn.Sequential(
                nn.Linear(Na, nnnet[0]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[0], nnnet[1]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[1], nnnet[2]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[2], nnnet[3]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[3], Ne)
                )                
        def forward(self, x):
            return self.layers(x)

    if (nngpu==0):  device   =  torch.device('cpu')
    else:           device   =  torch.device('cuda')
    model = MyNet()
    model.to(device)

    A = np.fromfile(file_absorbed, np.float32)[2:].reshape(cells, Na)
    E = np.fromfile(file_emitted,  np.float32)[2:].reshape(cells, Ne)

    # Drop parent cells (those have negative absorption value for the first frequency)
    m_leaf =  np.nonzero(A[:,0]>=0.0)
    A      =  A[m_leaf[0], :]
    E      =  E[m_leaf[0], :]
    
    cells  =  A.shape[0]
    
    dtype = torch.float
    xObs       =  torch.zeros((cells, Na), device=device, dtype=dtype)
    yObs       =  torch.zeros((cells, Ne), device=device, dtype=dtype)
    
    if (LOG):
        A = np.log10(np.clip(A, 1.0e-29, 1.0e32))
        E = np.log10(np.clip(E, 1.0e-29, 1.0e32))
    else:
        A = np.clip(A, 1.0e-29, 1.0e32)
        E = np.clip(E, 1.0e-29, 1.0e32)
        
    # NORMALISE, different factor for each frequency
    Ma = np.clip(np.mean(A, axis=0), 1.0e-20, 1.0e10)
    Me = np.clip(np.mean(E, axis=0), 1.0e-20, 1.0e10)
    print('Ma = ', Ma)
    print('Me = ', Me)
    sys.stdout.flush()
    for i in range(Na):  A[:,i] /= Ma[i]
    for i in range(Ne):  E[:,i] /= Me[i]
    np.asarray(Ma, np.float32).tofile(f'A_{dustname}.norm')
    np.asarray(Me, np.float32).tofile(f'E_{dustname}.norm')

    if (0):
        A.tofile('%s_A_fit.dump' % dustname)
        E.tofile('%s_E_fit.dump' % dustname)
    
    xObs[:,:]  =  torch.tensor(A[:,:], device=device, dtype=dtype)
    yObs[:,:]  =  torch.tensor(E[:,:], device=device, dtype=dtype)

    loss_function  =  nn.MSELoss()
    if (1):  optimizer  =  torch.optim.Adam(model.parameters(),  lr=0.0002) # 0.0003, 0.0001
    else:    optimizer  =  torch.optim.ASGD(model.parameters(),  lr=0.0003)

    MODEL = f'{prefix}_{dustname}.nn'
    try:
        model.load_state_dict(torch.load(MODEL))
        model.eval()
    except:
        print("Old result not reloaded")
        pass

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")    
    print('Training...', nnnet)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    
    sys.stdout.flush()
    train_steps   =  nnmaxiter
    t1 = time.time()    
    os.system('chroma.py 255 0 0')
    for epoch in range(train_steps):
        t0              = time.time()
        current_loss    = 0.0
        outputs         = model(xObs)
        loss            = loss_function(outputs, yObs)
        loss.backward()
        optimizer.step()
        current_loss   += loss.item()
        optimizer.zero_grad()
        if (epoch%100==0):
            t0 = time.time()-t0
            print("epoch %5d   loss %9.6f   %8.4f s/epoch" % (1+epoch, current_loss, t0/100.0))
    # os.system('chroma.py 0 255 0')
    torch.save(model.state_dict(), MODEL)
    t1 = time.time()-t1
    print("@@ Training %.3f s,  %.2f epochs/s" % (t1, train_steps/t1))
    sys.stdout.flush()

    
    if (1):
        # compare E and Epred
        t0    = time.time()
        Epred = model(xObs).cpu().detach().numpy()
        Epred.shape = (cells, Ne)
        t0 = time.time()-t0
        print("Predictions in %.3f seconds" % t0)
        plt.close(1)
        plt.figure(1, figsize=(10,7))
        plt.subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.96, wspace=0.59, hspace=0.35)
        for i in range(Ne): # over predicted wavelengths
            if (i>=9): break
            ax = plt.subplot(3,3,1+i)
            plt.title("%d" % E.shape[0])
            # plot every 31st point, training value and prediction
            plt.plot(E[0::11,i], Epred[0::11,i], 'k.', alpha=0.1)
            plt.text(0.1, 0.88, r'$\rm %.1f \/ \mu m$' % f2um(nnemit[i]), transform=ax.transAxes)
            xx = plt.xlim()
            plt.plot(xx, xx, 'r-', lw=1)            
            # estimate stdev relative to the solution
            r  =  (E[:,i]-Epred[:,i])/(E[:,i]+1.0e-10)   # relative error
            no =  100                                    # calculate sigma for no intervals over E
            p  =  np.percentile(E[:,i], np.linspace(0.0, 100.0, no+1))   # samples along the x-axis = E
            x, y = np.zeros(no, np.float32), np.zeros(no, np.float32)
            for k in range(no):   # corresponding y-axis values = <r>
                x[k]  =  0.5*(p[k]+p[k+1])  #  ~ E
                m     =  np.nonzero((E[:,i]>p[k])&(E[:,i]<p[k+1]))
                if (len(m[0])<1): y[k]  =  0.0
                else:             y[k]  =  np.sqrt(sum(r[m]**2)/len(m[0]))
            plt.xlabel(r'$I_{\nu} \rm (train, \/ %.1f \/ \mu m)$' % f2um(nnemit[i]))
            plt.ylabel(r'$I_{\nu} \rm (NN   , \/ %.1f \/ \mu m)$' % f2um(nnemit[i]))
            #--------------------------------------------------------------------------------
            plt.cax = plt.twinx()
            plt.plot(x, 100.0*y, 'b-', label='rms')
            xx = plt.xlim()
            plt.plot(xx, [10.0, 10.0], 'c--', label='10 pc')
            # plot 1%, 5%, 10%
            plt.plot([x[1], x[5], x[10]], 100.0*np.asarray([y[1],y[5],y[10]]), 'r+', label='1,5,10-q ')
            plt.xlim(xx)
            plt.ylim(-0.9, 35.0)
            plt.ylabel(r'$\sigma \/ \/ \rm [%s]$' % r'\%')
            plt.legend()
        plt.savefig(f'nnfit_{dustname}.png')
    


        
def NN_solve(prefix, dustname, nnabs, nnemit, file_absorbed, file_emitted='', nngpu=1, nnnet=[13,17,13]):
    """
    Assuming there already exists a NN solution in <prefix>_<dustname>.nn,
    convert absorptions [CELLS, nnabs] into emission [CELLS, nnemit].
    If file_emitted=='', return emitted[CELLS, nnemit] directly, instead
    of writing to a file.
    NOTE: the absorptions in file_absorbed need to be divided by 
          FF (integration weights used in ASOC.py), because the NN
          mapping must be independent of the frequency grid
          (different when NN is fitted and when it is used for predictions)
          ... was that finally not true?
    Input:
        prefix   =  prefix of the stored NN file of weights
        dustname =  name of the current dust component
        nnabs    =  vector of absorbed frequencies
        nnemit   =  vector of emitted frequencies
        file_absorbed = absorptions scaled to correspond to absorptions of the
                        current dust component, divided by abundance
        file_emitted  = file for the predicted emission, nnemit frequencies
    Note:
        - all data on absorptions and emissions read to Python arrays
        - GPU memory limitations are overcome by splitting the actual
          NN calculation to smaller batches of cells
    """
    Na = len(nnabs)
    Ne = len(nnemit)
    
    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            if (len(nnnet)==2):
                self.layers = nn.Sequential(
                nn.Linear(Na, nnnet[0]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[0], nnnet[1]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[1], Ne)
                )
            elif (len(nnnet)==3):
                self.layers = nn.Sequential(
                nn.Linear(Na, nnnet[0]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[0], nnnet[1]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[1], nnnet[2]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[2], Ne)
                )                
            elif (len(nnnet)==4):
                self.layers = nn.Sequential(
                nn.Linear(Na, nnnet[0]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[0], nnnet[1]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[1], nnnet[2]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[2], nnnet[3]),
                nn.LeakyReLU(),
                nn.Linear(nnnet[3], Ne)
                )                
        def forward(self, x):
            return self.layers(x)

    if (nngpu==0):  device   =  torch.device('cpu')
    else:           device   =  torch.device('cuda')
    model = MyNet()
    model.to(device)

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print("NN_solve  %s  -->  %s" % (file_absorbed, file_emitted), nnnet)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    
    A       = np.fromfile(file_absorbed, np.float32)[2:]
    cells   = len(A)//Na
    A.shape = (cells, Na)

    m_leaf  = np.nonzero(A[:,0]>-99)
    if (LOG):
        A = np.log10(np.clip(A, 1.0e-29, 1.0e32))
    else:
        A = np.clip(A, 1.0e-29, 1.0e32)        

    # normalisation factors for absorptions and emissions
    Ma = np.fromfile(f'A_{dustname}.norm', np.float32)
    Me = np.fromfile(f'E_{dustname}.norm', np.float32)
    
    # to eliminate scaling factor FF (cf. TW/FF in ASOC.py)
    #FF_now = np.zeros(Na, np.float32)
    #for ifreq in range(Na):
    #    FF_now[ifreq]  =  nnabs[ifreq] / 1.0e20
    #    if (ifreq==0):           FF_now[ifreq] *= 0.5*(nnabs[1]-nnabs[0])
    #    else:
    #        if (ifreq==(Na-1)):  FF_now[ifreq] *= 0.5*(nnabs[Na-1]-nnabs[Na-2])
    #        else:                FF_now[ifreq] *= 0.5*(nnabs[ifreq+1]-nnabs[ifreq-1])
    #
    # NORMALISED, absorptions divided by Ma, including AD HOC 1e20 for absorptions going into NN
    # NORMALISATION A /= FF_now WAS NOT NEEDED AFTER ALL ?
    for ifreq in range(Na):  
        A[:,ifreq]  /=  Ma[ifreq] #### * FF_now[ifreq]
        if (1):
            # nnsolve => write results for all cells... although some might be parent cells
            print("#@ solve ifreq=%d, %.3e Hz, normalised absorptions <> = %12.4e,  norm %12.4e" %
                  (ifreq, nnabs[ifreq], np.mean(A[m_leaf[0],ifreq]), Ma[ifreq]))
            print(np.nonzero(~np.isfinite(A[m_leaf[0], ifreq])))
    del m_leaf

        
        
    MODEL = f'{prefix}_{dustname}.nn'
    try:
        model.load_state_dict(torch.load(MODEL))
        model.eval()
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")        
        print("Old NN model not loaded")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit()
        pass

    t0    = time.time()
    dtype = torch.float
    
    if (0): 
        # all cells in one go
        # xObs       =  torch.zeros((cells, Na), device=device, dtype=dtype)
        yObs     =  torch.zeros((cells, Ne), device=device, dtype=dtype)
        xObs     =  torch.tensor(A[:,:], device=device, dtype=dtype)
        # no extrapolation to negative intensities !
        Epred    =  model(xObs).cpu().detach().numpy()
    else:
        # split cells into batches
        Epred  =  np.zeros((cells, Ne), np.float32)
        batch  =  1000000
        xObs   =  torch.zeros((batch, Na), device=device, dtype=dtype)
        yObs   =  torch.zeros((batch, Ne), device=device, dtype=dtype)
        a =  0
        while (a<cells):
            b  =  min(a+batch, cells)
            print("  .... NN_solve cells %d - %d   (%d cells)" % (a, b, b-a))
            # xObs[0:(b-a),:]  =  torch.tensor(A[a:b,:], device=device, dtype=dtype)
            xObs[0:(b-a),:]  =  torch.from_numpy(A[a:b,:])
            Epred[a:b,:]     =  model(xObs).cpu().detach().numpy()[0:(b-a),:]
            a += batch
            
    if (LOG):
        Epred = 10.0**Epred
    Epred = np.clip(Epred, 0.0, 1.0e32)
    Epred.shape = (cells, Ne)
    t0 = time.time()-t0
    print("Predictions in %.3f seconds" % t0)

    if (0):
        A.tofile('%s_A_solve.dump' % dustname)
        Epred.tofile('%s_E_solve.dump' % dustname)
    
    # NORMALISED, NN was mapping absorbed*1e20/FF/Ma  <-> emitted/Me => return emitted
    for i in range(Ne):  Epred[:,i] *= Me[i]

    # print("NN_solve => normalised emission %d x %d = cells %d" % (Epred.shape[0], Epred.shape[1], cells))
    
    if (len(file_emitted)<1):
        return Epred
    else:
        fp = open(file_emitted, 'wb')
        np.asarray([cells, Ne], np.int32).tofile(fp)
        np.asarray(Epred, np.float32).tofile(fp)
        fp.close()

    return None
