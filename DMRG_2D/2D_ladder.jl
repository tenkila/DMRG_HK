using ITensors, ITensorMPS, Statistics  # Import ITensor library and MPS/MPO functionality

# Construct the 2D Hubbard model Hamiltonian as an MPO for given twist phase in x-direction.
# `sites` is the site index set (with "Electron" site indices), and tx (complex) / ty (real) are hopping amplitudes 
# in the x and y directions respectively. U is the onsite repulsion.
function build_hubbard_mpo(sites, Lx, Ly, U, tx, ty)
    ampo = AutoMPO()  # AutoMPO for building sum of operators
    
    # We index sites linearly as i = (x-1)*Ly + y for site (x,y).
    # This maps the 2D lattice to a 1D chain in column-major order.
    # Note: for better DMRG performance on 2D, one could use a snake-like ordering 
    # to keep neighbors closer in the chain, but we use a simple mapping for clarity.
    
    for x in 1:Lx
        for y in 1:Ly
            # Linear index for site (x, y)
            local_i = (x - 1) * Ly + y
            
            # Hopping in the x-direction (periodic boundary in x):
            # Determine the neighbor in +x direction (with wrap-around)
            neighbor_x = (x == Lx ? 1 : x + 1)
            local_jx = (neighbor_x - 1) * Ly + y
            # Add hopping terms for spin-up and spin-down (tx may be complex)
            # Use tx for i->j and conj(tx) for j->i to ensure Hermitian conjugate
            ampo += -tx,        "Cdagup", local_i, "Cup",  local_jx   # c†_{i↑} c_{j↑} with phase
            ampo += -conj(tx),  "Cdagup", local_jx, "Cup", local_i    # (Hermitian conjugate)
            ampo += -tx,        "Cdagdn", local_i, "Cdn",  local_jx   # c†_{i↓} c_{j↓}
            ampo += -conj(tx),  "Cdagdn", local_jx, "Cdn", local_i    # (Hermitian conjugate)
            
            # Hopping in the y-direction (open boundary in y):
            if y < Ly
                # Neighbor in +y direction (no wrap, open boundary)
                local_jy = (x - 1) * Ly + (y + 1)
                # Add hopping terms for spin-up and spin-down (ty is real)
                ampo += -ty, "Cdagup", local_i, "Cup",  local_jy
                ampo += -ty, "Cdagup", local_jy, "Cup", local_i      # Hermitian conjugate (ty real ⇒ same coef)
                ampo += -ty, "Cdagdn", local_i, "Cdn",  local_jy
                ampo += -ty, "Cdagdn", local_jy, "Cdn", local_i
            end
            
            # On-site Hubbard interaction U * n_up * n_down:
            ampo += U, "Nupdn", local_i   # Nupdn operator counts a doubly occupied site (n_up * n_down)
	    #ampo += -U/2, "Nup", local_i   # Nup operator counts the number of spin-up electrons
	    #ampo += -U/2, "Ndn", local_i   # Ndn operator counts the number of spin-down electrons
        end
    end
    
    return MPO(ampo, sites)  # Convert the sum of operators into an MPO Hamiltonian
end

# Run DMRG to find the ground state energy of Hamiltonian `H` (given as an MPO).
# `sites` is the site index set (needed to construct initial MPS).
# Returns the ground state energy E0.
function run_dmrg(H::MPO, sites)
    # Prepare an initial MPS at half-filling (one electron per site).
    # We alternate spin-up and spin-down on neighboring sites to have total N = Lx*Ly and roughly zero total Sz.
    Nsites = length(sites)
    init_state = [isodd(i) ? "Up" : "Dn" for i in 1:Nsites]   # e.g., "Up","Dn","Up","Dn",...
    # (Each site has one particle: this fixes total particle number = Nsites, i.e. half-filling)
    psi0 = MPS(sites, init_state)
    
    # Set DMRG sweep parameters: number of sweeps, max bond dimensions, and truncation cutoff.
    # (These can be adjusted for convergence as needed.)
    nsweeps = 20
    maxdim = [100, 200, 400, 800, 1600, 3200]   # increasing max bond dimension per sweep
    cutoff = 1e-7                      # truncation cutoff for Schmidt values
    
    # Run the DMRG algorithm to find the ground state. 
    # Returns the ground state energy and optimized MPS (psi), but we only need the energy here.
    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    return energy
end

# Main function to compute the twist-averaged ground state energy.
# Lx, Ly: lattice dimensions; U: onsite repulsion; t: hopping amplitude (y-direction);
# k_num: number of k-points (twist angles) in the x-direction to average over.
function twist_average_energy(Lx, Ly, U, t, k_num)
    # Create site indices for spinful electrons on Lx*Ly sites, with quantum number conservation (conserve total N and Sz).
    sites = siteinds("Electron", Lx * Ly; conserve_qns=true)
    
    energies = Float64[]  # to collect ground state energies for each twist
    # Loop over kx values symmetrically around 0: -floor(k_num/2) to +floor(k_num/2)
    for kx in -div(k_num, 2):div(k_num, 2)
        # Compute the twist phase theta = (kx/k_num)*pi
        # and define complex hopping tx = t * e^(i * theta) for x-direction
        let theta = (kx / k_num) * pi/Lx
            tx = t * exp(im * theta)
            # Build the Hubbard Hamiltonian for this twist angle
            H = build_hubbard_mpo(sites, Lx, Ly, U, tx, t)
            # Find ground state energy with DMRG
            E0 = run_dmrg(H, sites)
            push!(energies, E0)
            @info "Twist kx=$kx: Ground state energy = $E0"   # log info for each twist (optional)
        end
    end
    
    # Compute the average of the energies over all twist angles
    avg_energy = mean(energies)
    println("Twist-averaged ground state energy: $avg_energy")
    return avg_energy
end

# Example usage:
#avgE = twist_average_energy(2, 4, 6.0, 1.0, 10)
#println("Average energy over twists: $avgE")

# Function to scan over different Lx values at fixed Ly, U, t, and k_num
function scan_Lx(Lx_list::Vector{Int}, Ly::Int, U::Float64, t::Float64, k_num::Int)
    energies = Float64[]  # list to collect twist-averaged energies
    
    for Lx in Lx_list
        println("\n--- Lx = $Lx ---")
        avg_energy = twist_average_energy(Lx, Ly, U, t, k_num)
	push!(energies, avg_energy/(Lx*Ly))
    end
    
    println("\nSummary:")
    println("Lx values: ", Lx_list)
    println("Twist-averaged energies: ", energies)
    
    return Lx_list, energies
end

# Example usage:
# Sweep over Lx = 2, 4, 6 at fixed Ly = 4, U = 8.0, t = 1.0, k_num = 5
Lx_list = [4, 6, 8, 10]
Ly = 2
U = 4.0
t = 1.0
k_num = 4

Lx_values, energy_values = scan_Lx(Lx_list, Ly, U, t, k_num)

