using Printf, Einsum, LinearAlgebra


a = [.5, .5]
b = [.5, .5]
M = [0. 1.; 1. 0.]
reg = 0.01


function sinkhorn(a,b,M,reg;numItermax=1000,
    stopThr=1e-9,verbose=false,log_euv=false)

    a = reshape(a,:,1)
    b = reshape(b,:,1)

    if length(a) == 0
        a = ones((size(M)[1],)) / size(M)[1]
    end
    if length(b) == 0
        b = ones((size(M)[2],)) / size(M)[2]
    end

    # init data
    Nini = length(a)
    Nfin = length(b)

    if length(size(b)) > 2
        nbb = size(b)[2]
    else
        nbb = 0
    end

    if log_euv
        log_dict = Dict("err" => [])
    end

    if nbb > 0
        u = ones((Nini, nbb)) / Nini
        v = ones((Nfin, nbb)) / Nfin
    else
        u = ones(Nini) / Nini
        v = ones(Nfin) / Nfin
    end

    # K = Array{Float64}(undef, size(M))
    K = map(exp,-M ./ reg)

    tmp2 = Array{Float64}(undef, size(b))

    Kp = reshape(1 ./ a, :, 1) .* K
    cpt = 0
    err = 1
    while (err > stopThr && cpt < numItermax)
        uprev = u
        vprev = v

        KtransposeU = K' * u
        v = b ./ KtransposeU
        u = 1 ./ (Kp * v)

        if (any(x->x==0,KtransposeU) ||
            any(isnan,u) || any(isnan,v) ||
            any(isinf,u) || any(isinf,v))

            @printf("Warning: numerical errors at iteration %.0f",cpt)
            u = uprev
            v = vprev
            break
        end
        if cpt % 10 == 0
            #check every ten steps faster
            if nbb > 0
                err = sum((u-uprev)^2) / sum(u^2) + \
                sum((v-vprev)^2) / sum(v^2)
            else
                @einsum tmp2[j] = u[i]*M[i,j]*v[j]
                err = norm(tmp2 - b)^2
            end
            if log_euv
                push!(log_dict["err"],err)
            end

            if verbose
                if cpt % 200 == 0
                    @printf("%.5s|%.12s\n","It.","Err")
                    println("-"^19)
                    @printf("%.5d|%.8e",cpt,err)
                end
            end
        end
        cpt += 1
    end
    if log_euv
        log_dict = merge(log_dict,Dict("u"=>u),Dict("v"=>v))
    end

    if length(nbb) > 1
        res = Array{Float64}(undef, size(v)[2])
        @einsum res[k] = u[i,k]*K[i,j]*v[j,k]*M[i,j]
        if log_euv
            return res, log_dict
        else
            return res
        end
    else #return OT matrix

        if log_euv
            return reshape(u,:,1) .* K .* reshape(v,:,1), log_dict
        else
            return reshape(u,:,1) .* K .* reshape(v,:,1)
        end
    end
end

 @time sinkhorn(a,b,M,0.01)
