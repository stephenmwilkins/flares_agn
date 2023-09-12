def get_lum(ii, tag, bins = np.arange(-26,-16,0.5), inp='FLARES', filter = 'FUV', LF = True, Luminosity='DustModelI'):
    if inp == 'FLARES':
        num = str(ii)
        if len(num) == 1:
            num =  '0'+num
        filename = rF"./data/flares.hdf5"
        num = num+'/'
    else:
        filename = F'./data/EAGLE_{inp}_sp_info.hdf5'
        num = ''
    with h5py.File(filename,'r') as hf:
        lum = np.array(hf[F"{num}/{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/{Luminosity}/{filter}"])
    if LF == True:
        tmp, edges = np.histogram(lum_to_M(lum), bins = bins)
        return tmp
    else: return lum

    
def get_lum_all(tag, bins = np.arange(-25, -16, 0.5), inp = 'FLARES', LF = True, filter = 'FUV', Luminosity='DustModelI'):
    if inp == 'FLARES':
        df = pd.read_csv('weight_files/weights_grid.txt')
        weights = np.array(df['weights'])
        sims = np.arange(0,len(weights))
        calc = partial(get_lum, tag = tag, bins = bins, inp = inp, LF = LF, filter = filter, Luminosity = Luminosity)
        #poiss_lims = partial(models.poisson_confidence_interval, p = 0.68)
        pool = schwimmbad.MultiPool(processes=12)
        dat = np.array(list(pool.map(calc, sims)))
        pool.close()
        if LF:
            hist = np.sum(dat, axis = 0)
            out = np.zeros(len(bins)-1)
            err = np.zeros(len(bins)-1)
            out_up = np.zeros(len(bins)-1)
            out_low = np.zeros(len(bins)-1)
            for ii, sim in enumerate(sims):
                out+=dat[ii]*weights[ii]
                err+=np.square(np.sqrt(dat[ii])*weights[ii])
            return out, hist, np.sqrt(err)
        else: return dat
    else:
        out = get_lum(00, tag = tag, bins = bins, inp = inp, filter = filter, Luminosity = Luminosity)
        return out
h = 0.6777
parent_volume = 3200**3
vol = (4/3)*np.pi*(14/h)**3
bins = -np.arange(16, 26.5, 0.5)[::-1]
bincen = (bins[1:]+bins[:-1])/2.
binwidth = bins[1:] - bins[:-1]
out, hist, err = get_lum_all('010_z005p000', bins=bins)
Msim = out/(binwidth*vol)
mask = np.where(hist==1)[0]
lolims = np.zeros(len(bincen))
lolims[mask] = True
y_lo = np.log10(Msim)-np.log10(Msim-yerr)
y_up =  np.log10(Msim+yerr)-np.log10(Msim)
y_lo[mask] = 0.5
y_up[mask] = 0.5
ok = np.where(hist<5)[0]
plt.errorbar(bincen, np.log10(Msim), yerr=[y_lo, y_up], lolims=lolims, ls='', marker='o')
plt.show()
