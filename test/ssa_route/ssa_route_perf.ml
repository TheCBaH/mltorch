(* The route gate's performance row for one model: the planned binary32 SSA
   kernels (the default performance policy) made through the direct frontend
   and through the structured one, compiled to two host binaries, run back to
   back in alternating order (Paired_timing: 3 warm-up pairs, 15 kept). The
   old form is Direct, the new one Structured. Passing means outputs bitwise
   equal in every pair and a decision other than Regression: the structured
   route is no slower, within the interval's resolution. Timings go to stdout
   and are not a golden.

   Usage: ssa_route_perf MODEL.pt2 *)

open Loop_ir
module G = Graph_ir
module Gate = Ssa_route_test.Route_gate

let now = Unix.gettimeofday

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  !acc

let () =
  let config =
    {
      Storage_script.Config.layout = Storage_script.Layout.Separate;
      constants = Storage_script.Ownership.Borrowed;
      inputs = Storage_script.Ownership.Borrowed;
    }
  in
  let b, constants = Model_load.load ~config Sys.argv.(1) in
  let numerics = Loop_numerics.Simd_fp32_relaxed in
  let vector = Loop_target.neon128 in
  let pipeline =
    Ssa_backends.Pipeline.Planned
      {
        numerics =
          Option.get (Ssa_ir.Ssa_numerics.of_name (Loop_numerics.name numerics));
        target =
          Ssa_ir.Ssa_target.with_row_block
            (Ssa_ir.Ssa_blocking.Rows.of_int vector.Loop_target.row_block)
            Ssa_ir.Ssa_target.neon128;
      }
  in
  let prepare frontend =
    let dir = Loop_c_exec.Proc.temp_dir "ssa_route_perf" in
    ( dir,
      Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
        (Loop_c_exec.Host.prepare ~vector ~numerics
           ~kernel:(Ssa_backends.c_via ~frontend ~pipeline) ~dir b
           ~constants:(fun id -> G.Tensor_id.Map.find_opt id constants)) )
  in
  let inputs = Gate.default_inputs b in
  let bind id = List.assoc_opt id inputs in
  let d_dir, direct = prepare Ssa_backends.Frontend.Direct in
  let s_dir, structured = prepare Ssa_backends.Frontend.Structured in
  Fun.protect
    ~finally:(fun () ->
      Loop_c_exec.Proc.remove_tree d_dir;
      Loop_c_exec.Proc.remove_tree s_dir)
    (fun () ->
      let sample p =
        let t0 = now () in
        let out =
          Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
            (Loop_c_exec.Host.run p ~bind)
        in
        (now () -. t0, List.map bits out)
      in
      let reference = snd (sample direct) in
      let pairs =
        List.init 18 (fun i ->
            let first_direct = i mod 2 = 0 in
            let a, ao = sample (if first_direct then direct else structured) in
            let c, co = sample (if first_direct then structured else direct) in
            let old_time, new_time, so =
              if first_direct then (a, c, co) else (c, a, ao)
            in
            {
              Paired_timing.old_time;
              new_time;
              output_matches = so = reference;
            })
      in
      match Paired_timing.decide pairs with
      | Error _ ->
          prerr_endline "ssa_route_perf: the pairs were rejected";
          exit 2
      | Ok r ->
          Fmt.pr "%s: %a@."
            (Filename.basename Sys.argv.(1))
            Paired_timing.pp_report r;
          if
            r.Paired_timing.decision = Paired_timing.Regression
            || r.Paired_timing.decision = Paired_timing.Output_differs
          then (
            Fmt.pr "gate: fail@.";
            exit 1)
          else Fmt.pr "gate: pass@.")
