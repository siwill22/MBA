import xarray as xr
from ptt.utils.call_system_command import call_system_command


def okabe(model_surface, rho, base_depth, observation_height=0):
    '''
    Wrapper function to call the GMT module 'grdgravmag3d', and 
    generate the gravity response of a grid of topography using 
    the method of Okabe (in which each grid cell is considered as a
    rectangular prism)
    '''
    


    model_surface.to_netcdf('../grids/test_bat.nc') 

    call_system_command(['/opt/anaconda3/envs/pygmt7/bin/gmt',
                         'grdgravmag3d',
                         '../grids/test_bat.nc',
                         '-G../grids/test_out.nc',
                         '-C{:f}'.format(rho),
                         '-Zb-{:f}'.format(base_depth),
                         '-L{:f}'.format(observation_height)])
    
    model = xr.load_dataarray('../grids/test_out.nc')

    return model

