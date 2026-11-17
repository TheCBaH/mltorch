(* The route gate's performance row for one model: the planned binary32 SSA
   kernels (the default performance policy) made through the direct frontend
   and through the structured one, compiled to two host binaries, run back to
   back in alternating order (Paired_timing: 3 warm-up pairs, 15 kept). The
   old form is Direct, the new one Structured. Passing means outputs bitwise
   equal in every pair and a decision other than Regression: the structured
   route is no slower, within the interval's resolution. Timings go to stdout
   and are not a golden.

   With a second argument [loop] the old form is instead the Loop emitter's
   kernel (Host.prepare's default, same numerics and target) and the new one
   the structured SSA kernel: the SSA-against-Loop row (make ssa.loop.perf),
   outputs compared within 1e-4 absolute and relative rather than bitwise.

   With [--kernels] the paired run is replaced by a per-invocation table: the
   generated model_infer.c of each form is rewritten to time every kernel call
   (a monotonic clock around the call alone, not the buffer initialisation),
   recompiled, and run with [--repeat] a few times alternating between the two
   forms; each invocation's time is the median over the warm runs of a
   process, then over the processes. Outputs of the timed binaries are checked
   against the old form under the same tolerance. The table lists the
   invocations by how much the new form loses, then the kernel-name classes.

   Usage: ssa_route_perf MODEL.pt2 [loop] [--kernels] *)

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

(* The Loop and SSA relaxed kernels may order a reduction differently, so in
   the loop row outputs are equal within atol = rtol = 1e-4 (the repo's
   --shadow-numeric tolerance), a NaN matching only a NaN. *)
let close a b =
  List.length a = List.length b
  && List.for_all2
       (fun x y ->
         let x = Int32.float_of_bits x and y = Int32.float_of_bits y in
         (Float.is_nan x && Float.is_nan y)
         || x = y
         || Float.abs (x -. y) <= 1e-4 +. (1e-4 *. Float.abs y))
       a b

(* ---- --kernels ---- *)

let scan l fmt f =
  try Some (Scanf.sscanf l fmt f)
  with Scanf.Scan_failure _ | Failure _ | End_of_file -> None

(* The kernel call of invocation [n] is [    if (NAME(error, ...)) {]; time it
   and keep the structure. Returns the text and the kernel name per position. *)
let instrument source =
  let names = Hashtbl.create 64 in
  let cur = ref (-1) and closing = ref false and in_run = ref false in
  let lines =
    List.concat_map
      (fun l ->
        if String.starts_with ~prefix:"int model_run(" l then in_run := true;
        (match scan l "  { /* invocation %d */" Fun.id with
        | Some n -> cur := n
        | None -> ());
        if String.starts_with ~prefix:"    if (" l && !cur >= 0 then (
          let call = String.sub l 8 (String.length l - 8 - 3) in
          Hashtbl.replace names !cur (String.sub call 0 (String.index call '('));
          closing := false;
          [
            Printf.sprintf
              "    { const double kt0 = kt_now(); const int kt_r = %s; \
               kt_ns[%d] = kt_now() - kt0; if (kt_r) {"
              call !cur;
          ])
        else if l = "      return 1;" then (
          closing := true;
          [ l ])
        else if !closing && l = "    }" then (
          closing := false;
          [ l; "    }" ])
        else if !in_run && l = "  return 0;" then (
          in_run := false;
          [ "  kt_commit();"; l ])
        else [ l ])
      (String.split_on_char '\n' source)
  in
  let n = Hashtbl.length names in
  let prelude =
    String.concat "\n"
      [
        "#define _POSIX_C_SOURCE 200809L";
        "#include <stdio.h>";
        "#include <stdlib.h>";
        "#include <time.h>";
        Printf.sprintf "#define KT_N %d" n;
        "#define KT_RUNS 256";
        "static double kt_ns[KT_N + 1];";
        "static double kt_s[KT_RUNS][KT_N + 1];";
        "static int kt_runs;";
        "static double kt_now(void) { struct timespec t; \
         clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + \
         t.tv_nsec; }";
        "static void kt_commit(void) { if (kt_runs < KT_RUNS) { for (int i = \
         0; i < KT_N; i++) kt_s[kt_runs][i] = kt_ns[i]; } kt_runs++; }";
        "__attribute__((destructor)) static void kt_dump(void) {";
        "  const char *path = getenv(\"KT_OUT\"); if (!path) return;";
        "  FILE *f = fopen(path, \"w\"); if (!f) return;";
        "  int runs = kt_runs < KT_RUNS ? kt_runs : KT_RUNS;";
        "  for (int i = 0; i < KT_N; i++) {";
        "    double v[KT_RUNS]; int m = 0;";
        "    for (int r = 1; r < runs; r++) {";
        "      double x = kt_s[r][i]; int j = m++;";
        "      while (j > 0 && v[j - 1] > x) { v[j] = v[j - 1]; j--; }";
        "      v[j] = x; }";
        "    fprintf(f, \"%d %.0f\\n\", i, m ? v[m / 2] : 0.0);";
        "  }";
        "  fclose(f); }";
        "";
      ]
  in
  (prelude ^ String.concat "\n" lines, names)

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a;
  a.(Array.length a / 2)

let kernels_table ~bind ~reference ~label_old ~label_new ~bundle ~ops old_p
    new_p =
  let rounds = 5 and repeat = 24 in
  let timed p =
    let dir = Loop_c_exec.Host.directory p in
    let src =
      Loop_c_exec.Proc.read_file (Filename.concat dir "model_infer.c")
    in
    let text, names = instrument src in
    let kt = Filename.concat dir "model_infer_kt.c" in
    Loop_c_exec.Proc.write_file kt text;
    let exe = Filename.concat dir "model_kt" in
    (match
       Loop_c_exec.Proc.run
         (Loop_c_exec.Host.compiler p
         @ [ Filename.concat dir "model_main.c"; kt; "-o"; exe; "-lm" ])
     with
    | Ok (Loop_c_exec.Proc.Exited 0, _) -> ()
    | Ok (_, log) -> failwith log
    | Error m -> failwith m);
    (exe, names)
  in
  let run (exe, _) p =
    let dir = Loop_c_exec.Host.directory p in
    let inputs = Filename.concat dir "kt_in.bin"
    and outputs = Filename.concat dir "kt_out.bin"
    and out = Filename.concat dir "kt_times.txt" in
    Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
      (Loop_c_exec.Host.write_inputs p ~bind ~path:inputs);
    (try Sys.remove outputs with Sys_error _ -> ());
    Unix.putenv "KT_OUT" out;
    let argv =
      [
        exe;
        "--weights";
        Loop_c_exec.Host.weights_path p;
        "--inputs";
        inputs;
        "--outputs";
        outputs;
        "--repeat";
        string_of_int repeat;
      ]
    in
    (match Loop_c_exec.Proc.run argv with
    | Ok (Loop_c_exec.Proc.Exited 0, _) -> ()
    | Ok (_, log) -> failwith log
    | Error m -> failwith m);
    let o =
      Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
        (Loop_c_exec.Host.read_outputs p ~path:outputs)
    in
    let times =
      String.split_on_char '\n' (Loop_c_exec.Proc.read_file out)
      |> List.filter_map (fun l -> scan l "%d %f" (fun i v -> (i, v)))
    in
    (List.concat_map bits o, times)
  in
  let a = timed old_p and b = timed new_p in
  let samples_a = ref [] and samples_b = ref [] and ok = ref true in
  for r = 0 to rounds - 1 do
    let go x p acc =
      let o, t = run x p in
      if not (close o reference) then ok := false;
      acc := t :: !acc
    in
    if r mod 2 = 0 then (
      go a old_p samples_a;
      go b new_p samples_b)
    else (
      go b new_p samples_b;
      go a old_p samples_a)
  done;
  let agg samples =
    let n = List.length (List.hd samples) in
    Array.init n (fun i -> median (List.map (fun t -> List.assoc i t) samples))
  in
  let ta = agg !samples_a and tb = agg !samples_b in
  let op_of i =
    let inv = List.nth bundle.Loop_bundle.invocations i in
    let name, detail = List.assoc inv.Loop_bundle.node ops in
    let one_line =
      String.concat " "
        (List.filter (( <> ) "")
           (String.split_on_char ' '
              (String.map (fun c -> if c = '\n' then ' ' else c) detail)))
    in
    let params =
      let key = "params=" in
      let n = String.length one_line and k = String.length key in
      let rec find i =
        if i + k > n then ""
        else if String.sub one_line i k = key then String.sub one_line i (n - i)
        else find (i + 1)
      in
      find 0
    in
    let outs =
      List.filter_map
        (fun (b : Loop_buffer.t) ->
          if b.Loop_buffer.role = Loop_buffer.Output then
            Some (Fmt.str "%a" Vec6.pp_shape b.Loop_buffer.sg.Tensor_sig.shape)
          else None)
        inv.Loop_bundle.program.Loop_program.buffers
    in
    (name, String.concat "," outs, params)
  in
  let total t = Array.fold_left ( +. ) 0. t in
  let pct x y = 100. *. (y -. x) /. Float.max x 1. in
  Fmt.pr "outputs of the timed binaries within tolerance: %b@." !ok;
  Fmt.pr "sum of kernel medians: %s %.0f ns, %s %.0f ns (%+.2f%%)@." label_old
    (total ta) label_new (total tb)
    (pct (total ta) (total tb));
  let rows =
    List.init (Array.length ta) (fun i -> (i, op_of i, ta.(i), tb.(i)))
  in
  Fmt.pr "@.worst invocations (inv, %s ns, %s ns, delta):@." label_old label_new;
  List.iteri
    (fun k (i, (na, sh, pa), x, y) ->
      if k < 25 then
        Fmt.pr "  %4d %10.0f %10.0f %+9.0f %+6.1f%%  %-12s %s@." i x y (y -. x)
          (pct x y) na
          (sh ^ " " ^ String.sub pa 0 (min 150 (String.length pa))))
    (List.sort
       (fun (_, _, x, y) (_, _, x', y') -> compare (y' -. x') (y -. x))
       rows);
  let classes = Hashtbl.create 16 in
  List.iter
    (fun (_, (na, _, pa), x, y) ->
      let c = na ^ " " ^ pa in
      let n, sx, sy =
        Option.value (Hashtbl.find_opt classes c) ~default:(0, 0., 0.)
      in
      Hashtbl.replace classes c (n + 1, sx +. x, sy +. y))
    rows;
  Fmt.pr "@.by class of the %s kernel name (count, %s ns, %s ns, delta):@."
    label_old label_old label_new;
  Hashtbl.fold (fun c v acc -> (c, v) :: acc) classes []
  |> List.sort (fun (_, (_, x, y)) (_, (_, x', y')) ->
      compare (y' -. x') (y -. x))
  |> List.iter (fun (c, (n, x, y)) ->
      Fmt.pr "  %4d %10.0f %10.0f %+9.0f %+6.1f%%  %s@." n x y (y -. x)
        (pct x y) c)

let () =
  let config =
    {
      Storage_script.Config.layout = Storage_script.Layout.Separate;
      constants = Storage_script.Ownership.Borrowed;
      inputs = Storage_script.Ownership.Borrowed;
    }
  in
  let b, constants, ops = Model_load.load_with_ops ~config Sys.argv.(1) in
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
  let flag f = Array.exists (String.equal f) Sys.argv in
  let against_loop = flag "loop" in
  let prepare frontend =
    let dir = Loop_c_exec.Proc.temp_dir "ssa_route_perf" in
    ( dir,
      Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
        (Loop_c_exec.Host.prepare ~vector ~numerics
           ?kernel:
             (match frontend with
             | None -> None
             | Some frontend -> Some (Ssa_backends.c_via ~frontend ~pipeline))
           ~dir b
           ~constants:(fun id -> G.Tensor_id.Map.find_opt id constants)) )
  in
  let inputs = Gate.default_inputs b in
  let bind id = List.assoc_opt id inputs in
  let d_dir, direct =
    prepare (if against_loop then None else Some Ssa_backends.Frontend.Direct)
  in
  let s_dir, structured = prepare (Some Ssa_backends.Frontend.Structured) in
  Fun.protect
    ~finally:(fun () ->
      if Sys.getenv_opt "SSA_PERF_KEEP" = None then (
        Loop_c_exec.Proc.remove_tree d_dir;
        Loop_c_exec.Proc.remove_tree s_dir)
      else Fmt.pr "kept: %s %s@." d_dir s_dir)
    (fun () ->
      let sample p =
        let t0 = now () in
        let out =
          Err.or_raise ~pp_error:Loop_c_exec.Host.pp_error
            (Loop_c_exec.Host.run p ~bind)
        in
        (now () -. t0, List.map bits out)
      in
      let reference = List.concat (snd (sample direct)) in
      let same o =
        if against_loop then close (List.concat o) reference
        else List.concat o = reference
      in
      if flag "--kernels" then
        kernels_table ~bind ~reference
          ~label_old:(if against_loop then "loop" else "direct")
          ~label_new:"structured" ~bundle:b ~ops direct structured
      else
        let pairs =
          List.init 18 (fun i ->
              let first_direct = i mod 2 = 0 in
              let a, ao =
                sample (if first_direct then direct else structured)
              in
              let c, co =
                sample (if first_direct then structured else direct)
              in
              let old_time, new_time, so =
                if first_direct then (a, c, co) else (c, a, ao)
              in
              { Paired_timing.old_time; new_time; output_matches = same so })
        in
        match Paired_timing.decide pairs with
        | Error _ ->
            prerr_endline "ssa_route_perf: the pairs were rejected";
            exit 2
        | Ok r ->
            Fmt.pr "%s%s: %a@."
              (Filename.basename Sys.argv.(1))
              (if against_loop then " (ssa vs loop)" else "")
              Paired_timing.pp_report r;
            if
              r.Paired_timing.decision = Paired_timing.Regression
              || r.Paired_timing.decision = Paired_timing.Output_differs
            then (
              Fmt.pr "gate: fail@.";
              exit 1)
            else Fmt.pr "gate: pass@.")
