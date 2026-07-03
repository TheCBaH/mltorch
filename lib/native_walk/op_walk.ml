module Pcg = Aten_spec.Pcg

let run (module M : Aten_walk_recipes.Walk.Op) ~ppf ~pcg ~steps =
  let run_step cfg pcg =
    let spec, pcg = M.build pcg cfg in
    (Aten_spec_run.run ~ppf spec, pcg)
  in
  let initial = M.cascade M.initial in
  Format.fprintf ppf "step 0: %a@." M.pp initial;
  let ok, pcg = run_step initial pcg in
  if ok then
    let rec loop i cfg pcg =
      if i > steps then ()
      else
        let axis, pcg = Aten_walk_recipes.Walk.pick pcg M.axes in
        let cfg, pcg = axis.Aten_walk_recipes.Walk.mutate pcg cfg in
        let cfg = M.cascade cfg in
        Format.fprintf ppf "step %d [%s]: %a@." i
          axis.Aten_walk_recipes.Walk.name M.pp cfg;
        let ok, pcg = run_step cfg pcg in
        if ok then loop (i + 1) cfg pcg
    in
    loop 1 initial pcg
