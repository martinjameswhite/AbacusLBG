#!/usr/bin/env python3
#
import numpy as np
import yaml
#
from astropy.table              import Table
from abacusnbody.hod.abacus_hod import AbacusHOD
from abacusnbody.metadata       import get_meta



class MockLBG:
    """A class to handle making mock LBG samples from halo catalogs."""
    # We may want to do a better job of holding the information in
    # this class.  Right now it's not very well thought through what is
    # kept internally vs. what is passed to each method.
    def __init__(self,yaml_file,mask_file,chi0):
        """Set up the class."""
        # Load the config file and parse in relevant parameters
        config          = yaml.safe_load(open(yaml_file))
        self.sim_params = config['sim_params']
        self.HOD_params = config['HOD_params']
        self.clustering_params = config['clustering_params']
        #
        # Get the metaparameters for the simulation.
        self.meta = get_meta(self.sim_params['sim_name'],\
                             redshift=self.sim_params['z_mock'])
        #
        # additional parameter choices -- we will apply RSD
        # ourselves so override the setting in HOD_params.
        self.want_rsd      = False
        self.write_to_disk = self.HOD_params['write_to_disk']
        #
        # Load the mask.
        self.msk = None
        #
        # Save some configuration parameters for later use.
        self.d = {}
        self.d['chi0'] = chi0
        self.d['zbox'] = self.meta['Redshift']
        self.d['abox'] = self.meta['ScaleFactor']
        self.d['Lbox'] = self.meta['BoxSizeHMpc']
        self.d['OmM' ] = self.meta['Omega_M']
        #
        # The velocity<->distance conversion for RSD,
        # assuming LCDM.
        OmM  = self.meta['Omega_M']
        Eofz = lambda zz: np.sqrt( OmM*(1+zz)**3+(1-OmM) )
        self.d['velf'] = self.d['abox']*100*Eofz(self.d['zbox'])
        #
        # Later we will want a random number generator.  Use a
        # fixed seed to make this reproducable.
        self.rng  = np.random.default_rng(1)
        #
    def periodic(self,pos):
        """Periodically wrap pos into -0.5Lbox to 0.5Lbox."""
        Lbox,Lbox2 = self.d['Lbox'],0.5*self.d['Lbox']
        wrap = np.nonzero( pos>=Lbox2 )[0]
        if len(wrap)>0: pos[wrap] -= Lbox
        wrap = np.nonzero( pos<-Lbox2 )[0]
        if len(wrap)>0: pos[wrap] += Lbox
        return(pos)
        #
    def set_hod(self,params):
        """Assign the HOD parameters."""
        # For the LRG HOD sigma is defined with natural logs,
        # with the sqrt{2}.
        # Satellite numbers are ncen times ([M-kappa.Mcut]/M1)^alpha
        for k in params.keys():
            self.HOD_params['LRG_params'][k] = params[k]
            self.d[k] = params[k]
        #
    def generate(self):
        """Calls abacusutils to generate the mock sample."""
        newBall = AbacusHOD(self.sim_params,\
                            self.HOD_params,self.clustering_params)
        lbgs = newBall.run_hod(newBall.tracers,self.want_rsd,\
                               self.write_to_disk,Nthread=16)
        # Generate a redshift, including peculiar velocity.
        lbgs['LRG']['zred'] = self.periodic(lbgs['LRG'][ 'z']+\
                                            lbgs['LRG']['vz']/self.d['velf'])
        # We want some statistics on this sample.
        self.d['nobj'] = lbgs['LRG']['mass'].size
        self.d['nbar'] = self.d['nobj']/self.d['Lbox']**3
        self.d['ncen'] = lbgs['LRG']['Ncent']
        self.d['fsat'] = 1-float(self.d['ncen'])/float(self.d['nobj'])
        # and save the sample for later processing.  Could instead
        # return this and then pass it to other methods.  TBD.
        self.lbgs  = lbgs
        # Finally let's make a bitmask for the objects.  If
        # we use a single byte FITS interprets this as a boolean,
        # so we'll use a 16-bit mask.
        self.bitmask = np.zeros(self.d['nobj'],dtype='uint16')
        #
    def assign_lum(self,bright_frac):
        """Assigns Llya to the mock objects."""
        # Eventually we could do a per-object luminosity using
        # abundance matching with scatter to a Lucy-deconvolved
        # Schechter LF.  For now all we care about is bright vs.
        # faint, which we do randomly.  We could also weight by
        # a power of halo mass.
        rr = self.rng.uniform(size=self.d['nobj'])
        self.bitmask[rr<bright_frac] |= 1
        #
    def select(self,diam,offset):
        """Select a small region of the box of diameter diam (radians)
        with the center of the box shifted by offset (fraction of box)."""
        # Also, at this point we should implement a z-dependent selection
        # function, but for now I'll use a hard zcut.
        Lbox  = self.d['Lbox']
        Lside = diam * self.d['chi0']
        depth = 0.8 * Lbox  # Most of the box, avoiding periodicity.
        gals  = self.lbgs['LRG']
        self.d['Lside'] = Lside
        # Shift the center of the box and select objects.
        xpos = self.periodic(gals['x'   ]+offset[0]*Lbox)
        ypos = self.periodic(gals['y'   ]+offset[1]*Lbox)
        zpos = self.periodic(gals['zred']+offset[2]*Lbox)
        in_survey = np.nonzero( (xpos>-0.5*Lside)&\
                                (xpos< 0.5*Lside)&\
                                (ypos>-0.5*Lside)&\
                                (ypos< 0.5*Lside)&\
                                (zpos>-0.5*depth)&\
                                (zpos< 0.5*depth)  )[0]
        # Store the results for later use.
        self.d['nkeep'] = len(in_survey)
        self.xpos = xpos[in_survey]
        self.ypos = ypos[in_survey]
        self.zpos = zpos[in_survey]
        self.bitm = self.bitmask[in_survey]
        #
    def make_hdr(self):
        """Puts some information into a 'header' dictionary."""
        hdr = {}
        hdr['sim'] = self.sim_params['sim_name']
        for k in self.d.keys(): hdr[k] = self.d[k]
        return(hdr)
        #
    def write_cat(self,outfn):
        """Write a catalog of objects in the survey."""
        # Generate the angular coordinates, in degrees.
        # Assume small angles and plane projection.
        ichi = 1.0/self.d['chi0'] * 180./np.pi
        rra  = self.xpos*ichi
        dec  = self.ypos*ichi
        chi  = self.zpos + self.chi0
        # and save them in a dictionary.
        hdr,outdict    = self.make_hdr(),{}
        hdr['COMMENT'] = 'Distances in Mpc/h'
        outdict['RA' ] = rra.astype('float32')
        outdict['DEC'] = dec.astype('float32')
        outdict['CHI'] = chi.astype('float32')
        outdict['BITMASK']=self.bitm
        tt = Table(outdict)
        for k in hdr.keys(): tt.meta[k] = hdr[k]
        tt.write(outfn,overwrite=True)
        #






if __name__=="__main__":
    diam   = 3.2 * np.pi/180.
    lbgs   = MockLBG('hod_big.yaml',None,4383.)
    params = {'logM_cut':12.15,'logM1':13.55,\
              'sigma':0.30,'kappa':1.00,'alpha':0.75}
    lbgs.set_hod(params)
    lbgs.generate()
    lbgs.assign_lum(0.5)
    lbgs.select(diam,[0.,0.,0.])
    lbgs.write_cat("mock_lbg_cat.fits")
    #
