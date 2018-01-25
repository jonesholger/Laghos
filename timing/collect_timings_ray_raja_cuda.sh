#!/usr/bin/env bash

options=( 'pa' 'fa' )

problem=0
parallel_refs=0
maxL2dof=8000000
nproc=4

outfile=rzmanta_timings_2d_cuda
mesh_file=../../data/square01_quad.mesh

calc() { awk "BEGIN{print $*}"; }

run_case()
{
    # Pass command as all inputs
    # Outputs: order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate
    "$@" | awk '
BEGIN { ref= 0 }
/--refine-serial/ { ref += $2 }
/--refine-parallel/ { ref += $2 }
/--order/ { order = $2 }
/Number of kinematic/ { h1_dofs = $7 }
/Number of specific internal energy/ { l2_dofs = $7 }
/CG \(H1\) rate/ { h1_cg_rate = $9 }
/CG \(L2\) rate/ { l2_cg_rate = $9 }
/Forces rate/ { forces_rate = $8 }
/UpdateQuadData rate/ { update_quad_rate = $8 }
/Major kernels total time/ { total_time = $6 }
END { printf("%d %d %d %d %.8f %.8f %.8f %.8f %.8f\n", order, ref, h1_dofs, l2_dofs, h1_cg_rate, l2_cg_rate, forces_rate, update_quad_rate, total_time) }'
}

[ -r $outfile ] && cp $outfile $outfile.bak
echo "# H1order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate total_time" > $outfile"_"${options[0]}
echo "# H1order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate total_time" > $outfile"_"${options[1]}
for method in "${options[@]}"; do
  for torder in {1..4}; do
    for sref in {0..12}; do
       nzones=$(( 4**(sref+1) ))
       nL2dof=$(( nzones*(torder+1)**2 ))
       if (( nproc <= nzones )) && (( nL2dof < maxL2dof )) ; then
         echo "np"$nproc "Q"$((torder+1))"Q"$torder $sref"ref" $method
         echo  $(run_case mpirun -np $nproc mpibind ./laghos.exe -$method \
                       -p $problem -tf 0.5 -cfl 0.05 -vs 1 \
                       --cg-tol 0 --cg-max-steps 50 \
                       --max-steps 1 \
                       --mesh $mesh_file \
                       --refine-serial $sref \
                       --refine-parallel $parallel_refs \
                       --order-thermo $torder \
                       --cuda \
                       --order-kinematic $((torder+1))) >> $outfile"_"$method
      fi
    done
  done
done
