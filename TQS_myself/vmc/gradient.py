import numpy as np



def vmc_gradient(
        model,
        samples,
        local_energies
):

    Oa=[]
    Ob=[]
    OW=[]


    for s in samples:

        oa,ob,ow=model.log_derivatives(s)

        Oa.append(oa)
        Ob.append(ob)
        OW.append(ow)



    Oa=np.array(Oa)
    Ob=np.array(Ob)
    OW=np.array(OW)


    E=np.mean(local_energies)


    grad_a=2*np.real(
        np.mean(
            local_energies[:,None]*np.conj(Oa),
            axis=0
        )
        -
        E*np.mean(np.conj(Oa),axis=0)
    )


    grad_b=2*np.real(
        np.mean(
            local_energies[:,None]*np.conj(Ob),
            axis=0
        )
        -
        E*np.mean(np.conj(Ob),axis=0)
    )


    grad_W=2*np.real(
        np.mean(
            local_energies[:,None,None]*np.conj(OW),
            axis=0
        )
        -
        E*np.mean(np.conj(OW),axis=0)
    )


    return grad_a,grad_b,grad_W