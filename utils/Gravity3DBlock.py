import numpy as np
import geopandas as gpd
from scipy.fftpack import fft2, ifft2
from GravityFFT import parkergrav

def glayer(rho, dx, z1, z2):
    """
    Calculate the vertical gravitational attraction on a two-dimensional grid caused by a two-dimensional density confined to a horizontal layer.
    
    Inputs:
        rho: A 2D array containing the two-dimensional density, in kg/(m**3).
        dx: Sample interval in the x direction, in km.
        z1: Depth to top of layer, in km. Must be > 0.
        z2: Depth to bottom of layer, in km. Must be > z1.
    
    Outputs:
        rho: Upon output, rho will contain the gravity anomaly, in mGal, with the same orientation as above.
    """
    si2mg = 1e5
    km2m = 1e3
    nx, ny = rho.shape
    dy = dx
    
    nn = [ny, nx]
    ndim = 2
    dkx = 2. * np.pi / (nx * dx)
    dky = 2. * np.pi / (ny * dy)
    
    store = fft2(rho)
    
    kx, ky = np.meshgrid(np.fft.fftfreq(ny, d=dy), np.fft.fftfreq(nx, d=dx))
    gf = gfilt(kx, ky, z1, z2)
    store = store * gf
    
    store = ifft2(store)
    rho = np.real(store) * si2mg * km2m
    
    return rho


def gfilt(kx, ky, z1, z2):
    """
    Calculate the value of the gravitational earth filter at a single (kx, ky) location.
    
    Inputs:
        kx: The wavenumber coordinate in the kx direction, in units of 1/km.
        ky: The wavenumber coordinate in the ky direction, in units of 1/km.
        z1: The depth to the top of the layer, in km.
        z2: The depth to the bottom of the layer, in km.
    
    Output:
        gf: The value of the earth filter.
    """
    gamma = 6.67e-11
    k = np.sqrt(kx**2 + ky**2)
    gf = (2 * np.pi * gamma * (np.exp(-k * z1) - np.exp(-k * z2))) / k
    gf[k == 0] = 2 * np.pi * gamma * (z2 - z1)
    
    return gf


'''
def kvalue(data, dx, dy=None):
    """
    Find the wavenumber coordinates of one element of a rectangular grid.
    
    Inputs:
        data: The input data array (not used in the calculation).
        dx: Sample interval in the x direction.
        dy: Sample interval in the y direction. If not provided, dy will be set to dx.
    
    Outputs:
        kx: The wavenumber coordinate in the x direction.
        ky: The wavenumber coordinate in the y direction.
    """
    ny, nx = data.shape
    if dy is None:
        dy = dx
    
    i = np.arange(1, ny+1)
    j = np.arange(1, nx+1)
    dkx = (2 * np.pi) / (nx * dx)
    dky = (2 * np.pi) / (ny * dy)

    nyqx = nx // 2 + 1
    nyqy = ny // 2 + 1

    ind = j < nyqx
    kx = np.zeros_like(j, dtype=float)
    kx[ind] = (j[ind] - 1) * dkx
    ind = j >= nyqx
    kx[ind] = (j[ind] - nx - 1) * dkx

    ind = i < nyqy
    ky = np.zeros_like(i, dtype=float)
    ky[ind] = (i[ind] - 1) * dky
    ind = i >= nyqy
    ky[ind] = (i[ind] - ny - 1) * dky

    return kx, ky
'''



def grav3dblock(x, y, srf1, srf2, layers, phi, c, rhoBackground):
    """
    Calculate the gravity effect of seawater and sediment layers.
    
    Inputs:
        x: X coordinates for grid/3D model, in km.
        y: Y coordinates for grid/3D model, in km.
        srf1: Bathymetry grid, when the water depth is 10, the value is 10, in km.
        srf2: Basement/sediment bottom, when its depth is 10, the value is 10, in km.
        layers: Surface of each layer in the 3D model, in km.
        phi: Parameter for the relationship between sediment porosity and buried depth.
        c: Parameter for the relationship between sediment porosity and buried depth.
        rhoBackground: Background density, in kg/m^3.
    
    Outputs:
        grvtotal: The gravity effect of seawater and sediment layer from free air anomaly.
        vm: Density of each layer, in kg/m^3.
    """
    m, n = srf1.shape
    dx = np.abs(x[1] - x[0])
    
    if rhoBackground is None:
        rhoBackground = 0
    
    vx, vy, vz = np.meshgrid(x, y, layers[:-1])
    vm = np.zeros_like(vx)
    
    print('gridding model in 3D...')
    
    # For each vertical profile within the cube, use logicals to determine
    # whether each depth point is below each surface and assign the density accordingly
    for mapX in range(n):
        for mapY in range(m):
            # force all cells above bathymetry to have water density
            ind = np.where(layers <= srf1[mapY, mapX])
            vm[mapY, mapX, ind] = rhoBackground - 1030

            # sediment density
            ind = np.where((layers > srf1[mapY, mapX]) & (layers < srf2[mapY, mapX]))
            if ind[0].size != 0:
                ZBelowBathy = layers[ind] - srf1[mapY, mapX]
                rhoS = DepthDependentDensity(ZBelowBathy * 1000, phi, c)
                rhoS = rhoBackground - rhoS
                vm[mapY, mapX, ind] = rhoS
        
        #print('done grav block column', mapX)
    
    print('starting summation of layers')
    
    # summing gravity effect of each layer in the model
    grvtotal = np.zeros((m, n))
    for layerindex in range(len(layers) - 1):
        z1 = layers[layerindex]
        z2 = layers[layerindex + 1]
        grvlayer = glayer(vm[:, :, layerindex], dx, z1, z2)
        
        grvtotal += grvlayer
        
        #print('done layer', layerindex)
    
    return grvtotal, vm



def DepthDependentDensity(z, phi0, c, RHOsg=2650, RHOw=1030):
    """
    Calculate the depth-dependent density based on the given parameters.

    Inputs:
        z: Depth values in meters.
        phi0: Parameter for the relationship between sediment porosity and buried depth.
        c: Parameter for the relationship between sediment porosity and buried depth.
        RHOsg: Optional. Sediment grain density in kg/m^3. Default value is 2650.
        RHOw: Optional. Water density in kg/m^3. Default value is 1030.

    Output:
        rhoS: Depth-dependent density in kg/m^3.
    """
    rhoS = (phi0 * np.exp(-c * z) * RHOw) + ((1 - (phi0 * np.exp(-c * z))) * RHOsg)
    
    return rhoS



def AddThermalAndPressureCorrection(X, Y, AgeGrid, proj, CrustThickness, ZMoho, gt, filePathForTransitionArea=None):
    """
    Add the gravity effect of density change by thermal expansion at the lithospheric mantle (125km depth to Moho).

    Inputs:
        X: X coordinates for the grid/3D model, in km.
        Y: Y coordinates for the grid/3D model, in km.
        AgeGrid: Oceanic crustal age. Its value is null at the continent.
        proj: Projection information. For example, "Mercator".
        CrustThickness: Crust thickness from the normal gravity inversion (removing the gravity effect of seawater and sediment).
        ZMoho: Moho interface grid, in km.
        gt: Residual gravity anomaly by conventional methods (constant mantle density).
        filePathForTransitionArea: Optional. File path for the ESRI shapefile representing the boundary of the transition area.
                                   The crustal age in this area will be interpolated linearly from continent age to oceanic age.

    Output:
        Gmra: Mantle residual gravity anomaly after adding lithospheric mantle thermal correction, in mGal.
    """

    # Beta grid
    BetaGrid = 30.0 / CrustThickness
    BetaGrid[BetaGrid < 1] = 1
    BetaGrid[~np.isnan(AgeGrid)] = 1000

    # LITHOSPHERE THERMAL GRAVITY CORRECTION
    if filePathForTransitionArea is not None:
        s = gpd.readfile(filePathForTransitionArea)
        iAgeGrid = MakeRegionalOCTGrid(X * 1000, Y * 1000, AgeGrid, s, 300)
        iAgeGrid = han3(iAgeGrid, 3)
    else:
        AgeGrid[np.isnan(AgeGrid)] = 300
        iAgeGrid = AgeGrid

    # Lithospheric mantle density
    rho = 3300

    # Create the surface for each layer of the lithospheric mantle (from 125km depth to Moho surface)
    layersNum = 20
    m, n = ZMoho.shape
    z = np.zeros((m, n, layersNum))
    z[:, :, 0] = ZMoho - np.nanmin(ZMoho) + 2

    for layerindex in range(1, layersNum):
        deltaZ = (125 - z[:, :, 0]) / layersNum
        z[:, :, layerindex] = z[:, :, layerindex - 1] + deltaZ

    # Add thermal correction
    grv, Tz, P, Rho = ThermalAndPressureGravityAnomaly(X, Y, z, BetaGrid, iAgeGrid, layersNum, rho)
    deltaRho = Rho - rho

    # Add thermal correction to gravity data
    Gmra = gt - grv

    return Gmra



def MakeRegionalOCTGrid(X, Y, AgeGrid, s, continentAge):
    """
    Invert crustal thickness/Moho depth with different background density and reference depth,
    and get the RMS value for each background density and reference depth.
    The RMS will be used for choosing the best parameters for inversion.

    Inputs:
        X: X coordinates for grid/3D model, in km.
        Y: Y coordinates for grid/3D model, in km.
        AgeGrid: Oceanic crustal age. Its value is null at the continent.
        s: The boundary of the transition area.
        continentAge: The age of the continent. It will be aligned to the continent at the final age grid result (iAgeGrid).

    Output:
        iAgeGrid: Crustal age grid after adding continental crustal age at the continent area and adding age at the transition area.
    """

    Xg, Yg = np.meshgrid(X, Y)

    # Age
    Continent0TransitionGrid1Ocean2 = np.ones_like(Xg)

    pX = s[1].X
    pY = s[1].Y
    in_ = inpolygon(Xg, Yg, pX, pY)
    Continent0TransitionGrid1Ocean2 = Continent0TransitionGrid1Ocean2 + in_ * 1

    pX = s[0].X
    pY = s[0].Y
    in_ = inpolygon(Xg, Yg, pX, pY)
    Continent0TransitionGrid1Ocean2 = Continent0TransitionGrid1Ocean2 + in_ * 1

    Continent0TransitionGrid1Ocean2 = Continent0TransitionGrid1Ocean2 - 1

    AgeGrid[Continent0TransitionGrid1Ocean2 == 0] = continentAge
    AgeGrid[Continent0TransitionGrid1Ocean2 == 1] = np.nan

    GridX, GridY = np.meshgrid(X, Y)
    GridX = GridX[~np.isnan(AgeGrid)]
    GridY = GridY[~np.isnan(AgeGrid)]
    AgeGridPts = AgeGrid[~np.isnan(AgeGrid)]
    iAgeGrid = gridfit(GridX, GridY, AgeGridPts, X, Y)

    return iAgeGrid


def Forward_NoThermalAndPressureCorrection(FAAGrid, BathyGrid, SedThickGrid, X, Y, 
                                           rhoBackground = 2700, phi0=0.55, c=4.5e-4):
    """
    Remove the gravity effect of seawater and sediment layer from the free air anomaly.

    Inputs:
        FAAGrid: Free air gravity anomaly at the marine area, in mGal.
        BathyGrid: Bathymetry grid, in km.
        SedThickGrid: Sediment grid, in km.
        X: X coordinates for grid/3D model, in km.
        Y: Y coordinates for grid/3D model, in km.
        phi0: Parameter for the relationship between sediment porosity and buried depth (see Sawyer 1985).
        c: Parameter for the relationship between sediment porosity and buried depth (see Sawyer 1985).

    Outputs:
        gt: The result of removing gravity effect of seawater and sediment layer from free air anomaly.
        bsmt: Basement interface, in km.
    """

    # Deal with the wrong value in the sediment and bathymetry grid
    SedThickGrid[SedThickGrid < 0] = 0
    SedThickGrid[np.isnan(SedThickGrid)] = 0
    BathyGrid[BathyGrid > 0] = 0

    bsmt = -BathyGrid + SedThickGrid
    bsmt[BathyGrid == 0] = 0  # Focus on the marine area

    # For gravity inversion codes, need distance units to all be in km
    BathyGrid = BathyGrid / 1000
    bsmt = bsmt / 1000

    # GRAVITY EFFECT OF WATER AND SEDIMENTS
    # Create surface of each layer in the 3D model, in km
    layers = np.concatenate([np.arange(0, 14100, 50), np.arange(14100, 20100, 100), np.arange(21000, 51000, 1000)]) / 1000

    # Calculate the gravity effect of seawater and sediment layer from free air anomaly
    gt, mv = grav3dblock(X, Y, -BathyGrid, bsmt, layers, phi0, c, rhoBackground)

    # Result
    #gt = gt + FAAGrid

    return gt, mv



def ThermalAndPressureGravityAnomaly(X, Y, z, BetaGrid, AgeGrid, layersNum, rho):
    """
    Calculate the gravity effect of density change by thermal expansion and pressure-driven compressibility in the lithospheric mantle (from Moho to 125km depth).

    Inputs:
        X: X coordinates for grid/3D model, in km.
        Y: Y coordinates for grid/3D model, in km.
        z: Interface of each layer of 3D model, in km.
        BetaGrid: Beta grid (please check the main paper).
        AgeGrid: Crustal age grid after adding continental crustal age at the continent area and adding age at the transition area.
        layersNum: Number of layers contained by 3D model.
        rho: Background density.

    Outputs:
        grvtotal: The gravity effect of density change by thermal expansion and pressure-driven compressibility in the lithospheric mantle (from Moho to 125km depth).
        Tz: Temperature of each layer, in K.
        P: Pressure of each layer, in kPa (1000 Pa).
        deltaRhoVolume: Each layer's density change by thermal expansion and pressure-driven compressibility in the lithospheric mantle (from Moho to 125km depth).
    """

    dx = (np.abs(X[1] - X[0])) / 1000

    # Constants parameters
    tau = 62.8  # Lithosphere cooling thermal decay constant
    a = 125  # Lithosphere thickness
    Tm = 1333  # Base lithosphere temperature
    alpha = 3.28e-5  # Coefficient of thermal expansion

    G = 6.67e-11  # Gravitational constant

    # Parameters for thermal expansion coefficient as the linear function of temperature
    alpha0 = 2.832e-5
    alpha1 = 0.758e-8

    numtaylor = 8

    ny, nx = AgeGrid.shape

    grvtotal = np.zeros((ny, nx))
    deltaRhoVolume = np.zeros((ny, nx, layersNum - 1))
    Tz = np.zeros((ny, nx, layersNum - 1))
    P = np.zeros((ny, nx, layersNum - 1))

    # Calculate the temperature field
    for layerindex in range(layersNum - 1):
        # Calculate the gravity anomaly using summation of Parker flat layers
        # McKenzie, 1978
        ConstantTerm1 = 2 / np.pi
        NthSum = np.zeros((ny, nx))

        for n in range(1, numtaylor + 1):
            NthTerm = (((-1) ** (n + 1)) / n) * ((BetaGrid / (n * np.pi)) * np.sin((n * np.pi) / BetaGrid)) \
                      * np.exp(((-n ** 2) * AgeGrid) / tau) * np.sin((n * np.pi * (a - z[:, :, layerindex])) / a)
            NthSum = NthSum + NthTerm

        Tratio = 1 - (a - z[:, :, layerindex]) / a + ConstantTerm1 * NthSum
        Tz[:, :, layerindex] = Tratio * Tm + 273

    # Calculate the pressure field; gravity effect of therm and pressure
    g = 9.8
    toleraceD = 0.0001
    for layerindex in range(layersNum - 1):
        # The top surface of the target layer
        if layerindex == 0:
            zUp = (z[:, :, 0] - 0) / 3  # Assume the crust, sediment, seawater have the same weight with this tick of mantle
        else:
            zUp = z[:, :, layerindex - 1]

        # Temperature difference between two neighboring grids
        if layerindex == 0:
            deltaT = Tz[:, :, layerindex] - 273  # For the top layer
        else:
            deltaT = Tz[:, :, layerindex] - Tz[:, :, layerindex - 1]

        # Upper layer density which is on the top of the target layer
        if layerindex == 0:
            rho_top = np.ones((ny, nx)) * rho
        else:
            rho_top = deltaRhoVolume[:, :, layerindex - 1]

        # Pressure difference on the target layer
        deltaZ = z[:, :, layerindex] - zUp
        deltaP = g * deltaZ * rho_top  # 9.8 m/s^2 * km * kg/m^3 = 1000 Pa

        # Different methods for calculating alpha
        # alpha = alpha0 + alpha1 * (Tz[:, :, layerindex])  # Bouhifd et al. (1996)
        # T = Tz[:, :, layerindex]
        # alpha = 6e-10 * T**3 - 2e-6 * T**2 + 0.0039 * T + 1.727
        # alpha = alpha * 1e-5  # Kroll 2012

        # Calculating beta
        K = 127.97 - 0.0232 * (Tz[:, :, layerindex] - 300)  # Forsterite, Function 31, Kroll-2012, the result K is for GPa
        beta = 1e-6 / K
        # beta = 1e-3 * alpha / 3.98  # Bouhifd et al. (1996)

        # Calculate the density of the target layer iteratively
        rho_below = rho_top * (1 - alpha * deltaT + beta * deltaP)
        rho_lastStep = rho_top

        while np.max(np.abs(rho_lastStep - rho_below)) > toleraceD:
            rho_lastStep = rho_below
            deltaP = g * deltaZ * rho_below
            rho_below = rho_top * (1 - alpha * deltaT + beta * deltaP)

        # Final pressure difference and density of the target layer
        P[:, :, layerindex] = deltaP
        deltaRhoVolume[:, :, layerindex] = rho_below

        # Gravity effect of the density change by thermal and pressure effect on the target layer
        z1 = z[:, :, layerindex]
        z2 = z[:, :, layerindex + 1]

        grvlayer = parkergrav(X, Y, z1, z2, rho_below - rho)
        grvtotal = grvtotal - grvlayer

    # Pressure field of the whole 3D model
    for layerindex in range(1, layersNum - 1):
        P[:, :, layerindex] = P[:, :, layerindex] + P[:, :, layerindex - 1]

    return grvtotal, Tz, P, deltaRhoVolume



def han3(in_grid, npass=1):
    """
    Hanning filter for a grid.

    Args:
        in_grid: Input grid to be filtered.
        npass: Number of passes to apply the Hanning filter (default is 1).

    Returns:
        gfilt: Filtered grid.
    """

    if npass < 1:
        npass = 1

    out = in_grid.copy()

    for num in range(npass):
        m, n = in_grid.shape
        i = np.arange(0, m-2)
        j = np.arange(0, n-2)
        han = np.zeros((m-2, n-2))
        han[i, j] = (in_grid[i, j] + in_grid[i+1, j] + in_grid[i+2, j]
                     + in_grid[i, j+1] + 2 * in_grid[i+1, j+1] + in_grid[i+2, j+1]
                     + in_grid[i, j+2] + in_grid[i+1, j+2] + in_grid[i+2, j+2]) / 10

        out = np.zeros((m, n))
        out[i+1, j+1] = han[i, j]
        out[1:m-1, 0] = han[:, 1]
        out[1:m-1, n-1] = han[:, n-2]
        out[0, 1:n-1] = han[1, :]
        out[m-1, 1:n-1] = han[m-2, :]
        out[0, 0] = np.mean([out[0, 1], out[1, 0]])
        out[0, n-1] = np.mean([out[0, n-2], out[1, n-1]])
        out[m-1, 0] = np.mean([out[m-1, 1], out[m-2, 0]])
        out[m-1, n-1] = np.mean([out[m-1, n-2], out[m-2, n-1]])

        in_grid = out

    return out
