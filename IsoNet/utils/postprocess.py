
def FCC(volume1, volume2, phiArray=[0.0], invertCone=False):
    """
    Fourier conic correlation



    [M, N, P] = volume1.shape
    [zmesh, ymesh, xmesh] = np.mgrid[-M /
                                        2:M / 2, -N / 2:N / 2, -P / 2:P / 2]
    # # The below is for RFFT implementation which is faster but gives numerically different results that potentially affect resolution estimation, DO NOT USE.
    # # The above is consistent with other programs such as FREALIGN v9.11 and relion_postprocess.
    # [zmesh, ymesh, xmesh] = np.mgrid[-M//2+m[0]:(M-1)//2+1, -N//2+m[1]:(N-1)//2+1, 0:P//2+1]
    # zmesh = np.fft.ifftshift( zmesh )
    # ymesh = np.fft.ifftshift( ymesh )
    """
    rhomax = int(
        np.ceil(np.sqrt(M * M / 4.0 + N * N / 4.0 + P * P / 4.0)) + 1)
    # if xy_only:
    #   zmesh *= 0
    #   rhomax = int( np.ceil( np.sqrt( N*N/4.0 + P*P/4.0) ) + 1 )
    # if z_only:
    #   xmesh *= 0
    #   ymesh *= 0
    #   rhomax = rhomax = int( np.ceil( np.sqrt( M*M/4.0 ) ) + 1 )
    rhomesh = ne.evaluate(
        "sqrt(xmesh * xmesh + ymesh * ymesh + zmesh * zmesh)")
    phimesh = ne.evaluate("arccos(zmesh / rhomesh)")
    phimesh[M // 2, N // 2, P // 2] = 0.0
    phimesh = np.ravel(phimesh)


    # phiArray = np.deg2rad(phiArray)
    phiArray = ne.evaluate("phiArray * pi / 180.0")

    rhoround = np.round(rhomesh.ravel()).astype('int')  # Indices for bincount
    # rhomax = int( np.ceil( np.sqrt( M*M/4.0 + N*N/4.0 + P*P/4.0) ) + 1 )

    fft1 = np.ravel(np.fft.fftshift(np.fft.fftn(volume1))).astype('complex128')
    conj_fft2 = np.ravel(np.fft.fftshift(
        np.fft.fftn(volume2)).conj()).astype('complex128')

    # # RFFT implementation faster but gives numerically different results that potentially affect resolution estimation, DO NOT USE.
    # # The above is consistent with other programs such as FREALIGN v9.11 and relion_postprocess.
    # fft1 = np.ravel( np.fft.rfftn( volume1 ) )
    # conj_fft2 = np.ravel( np.fft.rfftn( volume2 ) ).conj()

    FCC_normed = np.zeros([rhomax, len(phiArray)])
    for J, phiAngle in enumerate(phiArray):

        if phiAngle == 0.0:
            fft1_conic = fft1
            conj_fft2_conic = conj_fft2
            rhoround_conic = rhoround
        else:
            conic = np.ravel(ne.evaluate(
                "phimesh <= phiAngle + ((abs(phimesh - pi)) <= phiAngle)"))
            if invertCone:
                conic = np.invert(conic)
            rhoround_conic = rhoround[conic]
            fft1_conic = fft1[conic]
            conj_fft2_conic = conj_fft2[conic]
        FCC = np.bincount(rhoround_conic, ne.evaluate(
            "real(fft1_conic * conj_fft2_conic)"))
        Norm1 = np.bincount(rhoround_conic, ne.evaluate(
            "real(abs(fft1_conic)) * real(abs(fft1_conic))"))
        Norm2 = np.bincount(rhoround_conic, ne.evaluate(
            "real(abs(conj_fft2_conic)) * real(abs(conj_fft2_conic))"))

        goodIndices = np.argwhere(ne.evaluate("(Norm1 * Norm2) > 0.0"))[:-1]
        a = FCC[goodIndices]
        b = Norm1[goodIndices]
        c = Norm2[goodIndices]
        FCC_normed[goodIndices, J] = ne.evaluate("a / sqrt( b * c ) ")

    return FCC_normed


def FSC(volume1, volume2, phiArray=[0.0]):
    # FSC is just a wrapper to FCC

    return FCC(volume1, volume2, phiArray=phiArray)


def FSC3D():
    pass

def apply_bfac():
    pass