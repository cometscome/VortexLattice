function call_coefficients_2nd(dr,order) #Set the coefficients of 2nd order derivatives
    if order == 2
        indexofcoffs = zeros(Int64,2,3)
        valueofcoffs = zeros(Float64,2,3)

        indexofcoffs[:,1] .= -1
        indexofcoffs[:,2] .= 0
        indexofcoffs[:,3] .= 1
        for i=1:dim
            valueofcoffs[i,1] = 1/dr[i]^2
            valueofcoffs[i,2] = -2/dr[i]^2
            valueofcoffs[i,3] = 1/dr[i]^2
        end

    elseif order ==4
        indexofcoffs = zeros(Int64,2,5)
        valueofcoffs = zeros(Float64,2,5)

        indexofcoffs[:,1] .= -2
        indexofcoffs[:,2] .= -1
        indexofcoffs[:,3] .= 0
        indexofcoffs[:,4] .= 1
        indexofcoffs[:,5] .= 2
        for i=1:dim
            valueofcoffs[i,1] = -1/(12*dr[i]^2)
            valueofcoffs[i,2] = 16/(12*dr[i]^2)
            valueofcoffs[i,3] = -30/(12*dr[i]^2)
            valueofcoffs[i,4] = 16/(12*dr[i]^2)
            valueofcoffs[i,5] = -1/(12*dr[i]^2)
        end
    else
        println("Error!: order should be 2 or 4. Now, order = ",order)
        exit()
    end

    return indexofcoffs,valueofcoffs

end

function call_coefficients_1st(dr,order) #Set the coefficients of 2nd order derivatives
    if order == 2
        indexofcoffs = zeros(Int64,2,2)
        valueofcoffs = zeros(Float64,2,2)

        indexofcoffs[:,1] .= -1
        indexofcoffs[:,2] .= 1
        for i=1:dim
            valueofcoffs[i,1] = -1/2dr[i]
            valueofcoffs[i,2] = 1/2dr[i]
        end

    elseif order ==4
        indexofcoffs = zeros(Int64,2,4)
        valueofcoffs = zeros(Float64,2,4)

        indexofcoffs[:,1] .= -2
        indexofcoffs[:,2] .= -1
        indexofcoffs[:,3] .= 1
        indexofcoffs[:,4] .= 2
        for i=1:dim
            valueofcoffs[i,1] = 1/(12dr[i])
            valueofcoffs[i,2] = -8/(12dr[i])
            valueofcoffs[i,3] = 8/(12dr[i])
            valueofcoffs[i,4] = -1/(12dr[i])
        end
    else
        println("Error!: order should be 2 or 4. Now, order = ",order)
        exit()
    end

    return indexofcoffs,valueofcoffs

end
