"""Interpolates emiss curve to 150"""
function interpolate_emiss(emissin)

end

abstract type Input

end

abstract type GeometryClass <: Input

end
"""Geometry types that have analytically solveable emissivity"""
abstract type ExactGeometry <: GeometryClass

end

struct HexSphere <: ExactGeometry
    l_d_ratio #real number, length/diameter
end

struct TriGroove <: ExactGeometry

end

struct HexCavity <: ExactGeometry

end

struct Spheroid  <: GeometryClass

end

struct LaserParams <: Input

end

const EmissivityCurve = AbstractVector{Emissivity}
#TODO add a length param in type signature (150)    

const Emissivity = Pair{Wavelength, EmissivityValue}

struct Wavelength 
#might want to take log. TODO: Find unit package; use microns
end

struct EmissivityValue  
    #(R \geq 0, \leq 1 (may need to be looser; maybe use sigmoid))
end


function emissivity_of(emiss_in::EmissivityCurve,g::HexSphere)
    """TODO make it pretty (pi, *, etc)"""
    
    D  = 1 #diameter of sphere
    R  = D/2 #radius of sphere    /\
    L  = l_d*D #width of hexagon |  | distance between |  |
    #                             \/
    s=sqrt(L^2/3) #length of one side of hexagon |
   
    As=4*pi*R^2 #surface area of sphere
    Ah=3/2*s*L #surface area of hexagon (side times base over two times six)
    A1=As+Ah #surface area of solid
   
    A_1=A1; #emitting surface
    A_2=Ah; #apparent area
    # K=1-F11-A_2/A_1;
    emiss_out=1/(1+A_2/A_1*(1/emiss_in-1));
     
end
    
function emissivity_of(emiss_in,g::TriGroove)

end

function emissivity_of(emiss_in,g::HexCavity)

end    

function emissivity_of(emiss_in,g::Spheroid)

end

function emissivity_of(emiss_in,g::LaserParams)

end
