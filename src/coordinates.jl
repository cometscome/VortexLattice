function periodic!(ix,iy,lattice)
    ix += ifelse(ix > lattice.meshes[1],-lattice.meshes[1],0)
    iy += ifelse(iy > lattice.meshes[2],-lattice.meshes[2],0)
    ix += ifelse(ix < 1,lattice.meshes[1],0)
    iy += ifelse(iy < 1,lattice.meshes[2],0)
    return ix,iy
end

function coordinate2index(ix,iy,lattice)
    return (iy-1)*lattice.meshes[1] + ix
end

function index2coordinate(i,lattice)
    Nx = lattice.meshes[1]
    Ny = lattice.meshes[2]
    ix = (i-1) % Nx+1
    iy = div(i-ix,Nx)+1
    return ix,iy
end
