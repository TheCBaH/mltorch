(* Each case's invocations on the VME core: cycles (mcycle around the kernel
   call) and instructions retired, the data the estimate's rates are fitted
   to. The simulator's wall seconds are the host's, not the core's, so they
   stay out of the golden: with CORAL_ESTIMATE_WALL naming a file, each run's
   cycles and wall seconds are appended there, the emulator model's data. *)

module T = Machine_model_test.Model_test
module Cases = Coral_estimate_cases
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral

let config =
  {
    C.Config.dir = Sys.getenv "CORALNPU_DATA";
    configuration = "vme-core-mini-axi";
    deadline = 3600.;
    flags = [ "--cycles=2000000000" ];
  }

(* For a waveform study (not the golden): [CORAL_ESTIMATE_ONLY] runs the cases
   of that name alone, and [CORAL_ESTIMATE_FLAGS] adds simulator flags, each
   [{case}] in them the case's name made a file name ([--trace_file=],
   [--trace_select=]). *)
let only = Sys.getenv_opt "CORAL_ESTIMATE_ONLY"

let config_for name =
  match Sys.getenv_opt "CORAL_ESTIMATE_FLAGS" with
  | None -> config
  | Some extra ->
      let file =
        String.map
          (fun ch ->
            match ch with
            | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' -> ch
            | _ -> '_')
          name
      in
      let subst f = Str.global_replace (Str.regexp_string "{case}") file f in
      {
        config with
        C.Config.flags =
          config.C.Config.flags
          @ List.map subst (String.split_on_char ' ' (String.trim extra));
      }

let wall_log name i (r : C.Run.t) =
  match Sys.getenv_opt "CORAL_ESTIMATE_WALL" with
  | None -> ()
  | Some path ->
      Out_channel.with_open_gen [ Open_append; Open_creat ] 0o644 path
        (fun oc ->
          Printf.fprintf oc "%s\t%s\t%d\t%Ld\t%Ld\t%.3f\n"
            config.C.Config.configuration name i r.C.Run.mcycle r.C.Run.minstret
            r.C.Run.wall)

let run (c : Cases.t) =
  let runs = ref [] and rams = ref [] and image = ref [] in
  let on_run (o : C.Run.t) = runs := o :: !runs in
  let on_image placed = image := placed in
  (* the buffers of each invocation the layout put outside the DTCM *)
  let on_program _ _ regions =
    let off =
      List.filter_map
        (fun (b, r) ->
          match List.assoc_opt r !image with
          | Some Coral_profile.Role.Dtcm | None -> None
          | Some _ -> Some (b : Ssa_ir.Ssa_id.Buffer.t :> int))
        regions
    in
    rams := List.sort_uniq compare off :: !rams
  in
  let home =
    match c.Cases.home with
    | Cases.Home.Default -> None
    | Cases.Home.Ram -> Some (fun _ -> Coral_layout.Home.Ram)
  in
  T.check
    ~route:
      (C.route ?home ~on_run ~on_image ~label:c.Cases.name
         (config_for c.Cases.name))
    ~on_program ~contractions:c.Cases.contractions
    ~bf16_pairs:c.Cases.bf16_pairs ?attention:c.Cases.attention
    ?layer:c.Cases.layer ~pipeline:(Cases.pipeline c) ~oracle:c.Cases.oracle
    ~same:T.same_portable ~calls:1 ~input:c.Cases.input
    ~constant:(fun sg -> T.in_format sg (T.values ~salt:3 sg))
    c.Cases.name
    (fun () -> c.Cases.graph);
  List.iteri
    (fun i ((r : C.Run.t), ram) ->
      wall_log c.Cases.name i r;
      Fmt.pr "  [%d] cycles %Ld instret %Ld%s@." i r.C.Run.mcycle
        r.C.Run.minstret
        (if ram = [] then ""
         else " ram " ^ String.concat "," (List.map string_of_int ram)))
    (List.combine (List.rev !runs) (List.rev !rams))

(* A view-call case: the banded program planned as the route plans it, run on
   the core, its output bitwise the SSA interpreter's. The inputs are the
   case's own deterministic values. *)
module View = struct
  open Ssa_ir
  open Machine_ir
  open Machine_interp
  open Machine_lower
  module Src = Machine_source_test.Mir_source

  let inputs (p : Ssa_program.t) =
    let lcg = ref 5 in
    let value () =
      lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
      F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 977.)
    in
    List.filter_map
      (fun (b : Ssa_buffer.t) ->
        match b.Ssa_buffer.role with
        | Ssa_buffer.Input ->
            let n =
              Int64.to_int
                (Option.get (Ssa_buffer.elements b.Ssa_buffer.extents))
            in
            Some ((b.Ssa_buffer.id :> int), Array.init n (fun _ -> value ()))
        | _ -> None)
      p.Ssa_program.buffers

  let bits x =
    Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xFFFF_FFFFL

  let reference (p : Ssa_program.t) ~inputs =
    let memory =
      List.fold_left
        (fun m (b : Ssa_buffer.t) ->
          let cells =
            match List.assoc_opt (b.Ssa_buffer.id :> int) inputs with
            | Some c -> c
            | None ->
                Array.make
                  (Int64.to_int
                     (Option.get (Ssa_buffer.elements b.Ssa_buffer.extents)))
                  0.
          in
          Ssa_id.Buffer.Map.add b.Ssa_buffer.id (Ssa_memory.Floats cells) m)
        Ssa_id.Buffer.Map.empty p.Ssa_program.buffers
    in
    Err.or_raise ~pp_error:Ssa_interp.pp_error (Ssa_interp.run p ~memory);
    List.filter_map
      (fun (b : Ssa_buffer.t) ->
        match
          (b.Ssa_buffer.role, Ssa_id.Buffer.Map.find b.Ssa_buffer.id memory)
        with
        | Ssa_buffer.Output, Ssa_memory.Floats cs -> Some (Array.map bits cs)
        | _ -> None)
      p.Ssa_program.buffers

  let run ?engine_mutation (v : Cases.View.t) =
    let p = Cases.View.program v in
    let inputs = inputs p in
    let want = reference p ~inputs in
    let cfg = config_for v.Cases.View.name in
    let p =
      match C.finalize cfg p with
      | Ok p -> p
      | Error e -> failwith ("finalize: " ^ e)
    in
    let planning =
      Mir_planning.make ~subject:(Mir_lower.subject p) ~policy:"reference_f64"
        ~schedule:"scalar" ~precision:Mir_planning.Precision.F32
        ~lanes:(Mir_type.Lanes.of_int 1) ~fma:Mir_planning.Fma.Forbidden
        ~capabilities:[]
    in
    let lowered =
      match Err.payload (Mir_lower.program ~planning:(Some planning) p) with
      | Ok l -> l
      | Error r -> Fmt.failwith "lower: %a" Mir_lower.Refusal.pp r
    in
    let runs = ref [] in
    let exec =
      match
        C.route
          ~on_run:(fun r -> runs := r :: !runs)
          ?engine_mutation ~label:v.Cases.View.name cfg
      with
      | Machine_model.Mir_model_route.Route.Custom c -> (
          match
            c.Machine_model.Mir_model_route.Custom.exec ~sites:[||]
              lowered.Mir_lower.program
          with
          | Ok e -> e
          | Error e -> failwith ("exec: " ^ e))
      | _ -> assert false
    in
    let memory = Mir_memory.create () in
    let binding =
      match
        exec.Machine_model.Mir_model_route.Exec.instantiate memory
          ~shared:(fun _ -> None)
      with
      | Ok b -> b
      | Error e -> failwith e
    in
    List.iter
      (fun (e : Mir_layout_map.Entry.t) ->
        let b = e.Mir_layout_map.Entry.buffer in
        match List.assoc_opt (b.Ssa_buffer.id :> int) inputs with
        | None -> ()
        | Some cells ->
            Mir_memory.write_string memory
              (Option.get
                 (Mir_interp.Binding.instance binding
                    e.Mir_layout_map.Entry.region))
              ~offset:0L
              (Src.pack_cells b (Ssa_memory.Floats cells)))
      lowered.Mir_lower.layout;
    (match
       exec.Machine_model.Mir_model_route.Exec.run memory binding ~invocation:0l
     with
    | Mir_observation.Status.Unsupported e -> Fmt.failwith "unsupported: %s" e
    | _ -> ());
    let got =
      List.map
        (fun (o : Mir_observation.Output.t) ->
          Array.map
            (function
              | Some (c : Mir_const.t) ->
                  Int64.logand c.Mir_const.bits 0xFFFF_FFFFL
              | None -> -1L)
            o.Mir_observation.Output.cells)
        (Src.mir_outputs memory binding lowered.Mir_lower.layout)
    in
    Fmt.pr "%s%s (1 invocations): %s@." v.Cases.View.name
      (if engine_mutation = None then "" else ", mutated")
      (if got = want then "bitwise" else "DIFFERS");
    if engine_mutation = None then
      List.iteri
        (fun i (r : C.Run.t) ->
          wall_log v.Cases.View.name i r;
          Fmt.pr "  [%d] cycles %Ld instret %Ld@." i r.C.Run.mcycle
            r.C.Run.minstret)
        (List.rev !runs)

  let all () =
    List.iter
      (fun (v : Cases.View.t) ->
        if
          match only with
          | None -> true
          | Some n -> String.starts_with ~prefix:n v.Cases.View.name
        then
          try run v
          with e ->
            Fmt.pr "%s: raised %s@." v.Cases.View.name
              (List.hd (String.split_on_char '\n' (Printexc.to_string e))))
      Cases.View.all
end

(* The cases in parallel: one forked worker per case, at most the CPU pool's
   size at once (each simulator also holds its own slots), each writing its
   report to a file of its own; the reports are printed in case order, so the
   golden does not depend on which finished first. *)
let parallel cases =
  let cases =
    match only with
    | None -> cases
    | Some n -> List.filter (fun (c : Cases.t) -> c.Cases.name = n) cases
  in
  let dir = Filename.temp_file "coral_calibrate_" "" in
  Sys.remove dir;
  Sys.mkdir dir 0o700;
  let file i = Filename.concat dir (string_of_int i) in
  let width = Coral_cpus.pool () in
  let running = ref [] in
  let reap () =
    let pid, _ = Unix.wait () in
    running := List.filter (fun p -> p <> pid) !running
  in
  Format.print_flush ();
  flush stdout;
  flush stderr;
  List.iteri
    (fun i c ->
      while List.length !running >= width do
        reap ()
      done;
      match Unix.fork () with
      | 0 ->
          let fd =
            Unix.openfile (file i)
              [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ]
              0o600
          in
          Unix.dup2 fd Unix.stdout;
          (try run c
           with e ->
             (* the first line: the rest is a backtrace *)
             Fmt.pr "%s: raised %s@." c.Cases.name
               (List.hd (String.split_on_char '\n' (Printexc.to_string e))));
          Format.print_flush ();
          flush stdout;
          Unix._exit 0
      | pid -> running := pid :: !running)
    cases;
  while !running <> [] do
    reap ()
  done;
  List.iteri
    (fun i _ ->
      print_string
        (try In_channel.with_open_bin (file i) In_channel.input_all
         with Sys_error _ -> "");
      try Sys.remove (file i) with Sys_error _ -> ())
    cases;
  try Sys.rmdir dir with Sys_error _ -> ()

let%expect_test "the fit set" =
  parallel Cases.fit;
  [%expect
    {|
    cos 16x16 (1 invocations): bitwise
      [0] cycles 23061 instret 10726
    cos 16x16, scalar (1 invocations): bitwise
      [0] cycles 72794 instret 40670
    cos 32x32 (1 invocations): bitwise
      [0] cycles 92037 instret 42790
    cos 32x32, scalar (1 invocations): bitwise
      [0] cycles 290659 instret 162135
    exp 16x16 (1 invocations): bitwise
      [0] cycles 12757 instret 6246
    exp 16x16, scalar (1 invocations): bitwise
      [0] cycles 48575 instret 28804
    exp 32x32 (1 invocations): bitwise
      [0] cycles 50821 instret 24870
    exp 32x32, scalar (1 invocations): bitwise
      [0] cycles 193727 instret 114688
    log 16x16 (1 invocations): bitwise
      [0] cycles 27845 instret 13622
    log 16x16, scalar (1 invocations): bitwise
      [0] cycles 108751 instret 54066
    log 32x32 (1 invocations): bitwise
      [0] cycles 111173 instret 54374
    log 32x32, scalar (1 invocations): bitwise
      [0] cycles 434482 instret 215722
    sin 16x16 (1 invocations): bitwise
      [0] cycles 22709 instret 10776
    sin 16x16, scalar (1 invocations): bitwise
      [0] cycles 71689 instret 40299
    sin 32x32 (1 invocations): bitwise
      [0] cycles 90629 instret 42984
    sin 32x32, scalar (1 invocations): bitwise
      [0] cycles 286229 instret 160653
    sqrt 16x16 (1 invocations): bitwise
      [0] cycles 1234 instret 486
    sqrt 16x16, scalar (1 invocations): bitwise
      [0] cycles 8035 instret 2552
    sqrt 32x32 (1 invocations): bitwise
      [0] cycles 4738 instret 1830
    sqrt 32x32, scalar (1 invocations): bitwise
      [0] cycles 31589 instret 9672
    add 16x64 (1 invocations): bitwise
      [0] cycles 2628 instret 2119
    add 16x64, scalar (1 invocations): bitwise
      [0] cycles 18690 instret 11562
    add 32x128 (1 invocations): bitwise
      [0] cycles 16452 instret 8359 ram 1,2
    add 32x128, scalar (1 invocations): bitwise
      [0] cycles 82402 instret 45610 ram 1,2
    div 16x64 (1 invocations): bitwise
      [0] cycles 4972 instret 2343
    div 16x64, scalar (1 invocations): bitwise
      [0] cycles 30861 instret 11562
    div 32x128 (1 invocations): bitwise
      [0] cycles 20694 instret 9255 ram 1,2
    div 32x128, scalar (1 invocations): bitwise
      [0] cycles 131098 instret 45610 ram 1,2
    mul 16x64 (1 invocations): bitwise
      [0] cycles 2628 instret 2119
    mul 16x64, scalar (1 invocations): bitwise
      [0] cycles 18690 instret 11562
    mul 32x128 (1 invocations): bitwise
      [0] cycles 16452 instret 8359 ram 1,2
    mul 32x128, scalar (1 invocations): bitwise
      [0] cycles 82402 instret 45610 ram 1,2
    x*2 16x64 (1 invocations): bitwise
      [0] cycles 1899 instret 1613
    x*2 16x64, scalar (1 invocations): bitwise
      [0] cycles 15592 instret 9470
    x*2 32x128 (1 invocations): bitwise
      [0] cycles 11340 instret 6317 ram 1
    x*2 32x128, scalar (1 invocations): bitwise
      [0] cycles 61833 instret 37326 ram 1
    x*x+x 16x64 (2 invocations): bitwise
      [0] cycles 2625 instret 1862
      [1] cycles 2628 instret 2119
    x*x+x 16x64, scalar (2 invocations): bitwise
      [0] cycles 15584 instret 9465
      [1] cycles 18690 instret 11562
    x*x+x 32x128 (2 invocations): bitwise
      [0] cycles 14401 instret 7334 ram 1
      [1] cycles 16452 instret 8359 ram 0,2
    x*x+x 32x128, scalar (2 invocations): bitwise
      [0] cycles 61825 instret 37321 ram 1
      [1] cycles 82402 instret 45610 ram 0,2
    x+y+z 16x64 (2 invocations): bitwise
      [0] cycles 2628 instret 2119
      [1] cycles 2628 instret 2119
    x+y+z 16x64, scalar (2 invocations): bitwise
      [0] cycles 18690 instret 11562
      [1] cycles 18690 instret 11562
    x+y+z 32x128 (2 invocations): bitwise
      [0] cycles 16452 instret 8359 ram 1,3
      [1] cycles 16452 instret 8359 ram 2,4
    x+y+z 32x128, scalar (2 invocations): bitwise
      [0] cycles 82402 instret 45610 ram 1,3
      [1] cycles 82402 instret 45610 ram 2,4
    add 16x64, in RAM (1 invocations): bitwise
      [0] cycles 4676 instret 2119 ram 0,1,2
    add 32x128, in RAM (1 invocations): bitwise
      [0] cycles 18500 instret 8359 ram 0,1,2
    div 16x64, in RAM (1 invocations): bitwise
      [0] cycles 5735 instret 2343 ram 0,1,2
    div 32x128, in RAM (1 invocations): bitwise
      [0] cycles 22741 instret 9255 ram 0,1,2
    exp 32x32, in RAM (1 invocations): bitwise
      [0] cycles 51845 instret 24870 ram 0,1
    add 16x64, scalar, in RAM (1 invocations): bitwise
      [0] cycles 22802 instret 11562 ram 0,1,2
    softmax over C 16x16 (1 invocations): bitwise
      [0] cycles 34255 instret 16820
    softmax over C 32x32 (1 invocations): bitwise
      [0] cycles 134651 instret 65594
    sum over C 16x64 (1 invocations): bitwise
      [0] cycles 1828 instret 1741
    sum over C 32x128 (1 invocations): bitwise
      [0] cycles 7040 instret 6762
    sum over W 16x64 (1 invocations): bitwise
      [0] cycles 1516 instret 1440
    transpose 32x32 (1 invocations): bitwise
      [0] cycles 3172 instret 2252
    transpose 64x64 (1 invocations): bitwise
      [0] cycles 15492 instret 8108 ram 1
    transpose 32x32, in RAM (1 invocations): bitwise
      [0] cycles 4708 instret 2252 ram 0,1
    head split 16x64, 2 heads (2 invocations): bitwise
      [0] cycles 173009 instret 108328
      [1] cycles 2423 instret 1848
    head split 32x128, 4 heads (2 invocations): bitwise
      [0] cycles 731737 instret 432548 ram 1
      [1] cycles 13450 instret 7151 ram 2
    head split 49x64, 2 heads (2 invocations): bitwise
      [0] cycles 567288 instret 337878 ram 1
      [1] cycles 10318 instret 5481 ram 2
    linear 9x32 k64 (1 invocations): bitwise
      [0] cycles 4844 instret 2145
    linear 16x64 k64 (1 invocations): bitwise
      [0] cycles 13922 instret 4945 ram 2,3
    linear 64x64 k64 (1 invocations): bitwise
      [0] cycles 57286 instret 19534 ram 1,2,3
    linear 16x128 k32 (1 invocations): bitwise
      [0] cycles 18302 instret 6785 ram 2,3
    linear 32x32 k128 (1 invocations): bitwise
      [0] cycles 23374 instret 7973 ram 1,3
    linear 48x80 k32 (1 invocations): bitwise
      [0] cycles 34395 instret 13360 ram 2,3
    linear bf16 9x32 k64 (1 invocations): bitwise
      [0] cycles 4191 instret 1800
    linear bf16 16x64 k64 (1 invocations): bitwise
      [0] cycles 10403 instret 3965 ram 2,3
    linear bf16 64x64 k64 (1 invocations): bitwise
      [0] cycles 28186 instret 11662 ram 1,2,3
    linear bf16 32x128 k32 (1 invocations): bitwise
      [0] cycles 23631 instret 9943 ram 2,3
    linear bf16 48x48 k48 (1 invocations): bitwise
      [0] cycles 16482 instret 7673 ram 2,3
    sdpa bf16 32x32 d32 (1 invocations): bitwise
      [0] cycles 87661 instret 30256
    sdpa bf16 64x64 d64 (1 invocations): bitwise
      [0] cycles 373801 instret 127358 ram 0,1,2,8
    sdpa bf16 16x48 d32 (1 invocations): bitwise
      [0] cycles 66169 instret 23479
    sdpa bf16 16x16 d32 bk16 (1 invocations): bitwise
      [0] cycles 25113 instret 8690
    sdpa bf16 32x64 d64 bk16 (1 invocations): bitwise
      [0] cycles 202773 instret 72454 ram 1,2,8
    sdpa bf16 64x32 d32 (1 invocations): bitwise
      [0] cycles 175305 instret 60394 ram 8
    linear 9x32 k64, in RAM (1 invocations): bitwise
      [0] cycles 5853 instret 2145 ram 0,1,2,3
    linear 32x32 k64, in RAM (1 invocations): bitwise
      [0] cycles 15120 instret 5157 ram 0,1,2,3
    linear bf16 16x256 k64 (1 invocations): bitwise
      [0] cycles 52692 instret 15385 ram 1,3
    linear bf16 32x128 k128 (1 invocations): bitwise
      [0] cycles 95847 instret 23269 ram 1,2,3
    linear bf16 49x512 k64 (1 invocations): bitwise
      [0] cycles 391657 instret 111523 ram 1,2,3
    linear bf16 16x256 k64, in RAM (1 invocations): bitwise
      [0] cycles 58806 instret 15385 ram 0,1,2,3
    linear bf16 16x64 k64, in RAM (1 invocations): bitwise
      [0] cycles 10595 instret 3965 ram 0,1,2,3
    linear bf16 64x64 k64, in RAM (1 invocations): bitwise
      [0] cycles 28314 instret 11662 ram 0,1,2,3
    sdpa bf16 32x32 d32, in RAM (1 invocations): bitwise
      [0] cycles 88109 instret 30256 ram 0,1,2,7,8
    sdpa bf16 16x16 d32 bk16, in RAM (1 invocations): bitwise
      [0] cycles 25273 instret 8690 ram 0,1,2,7,8
    sdpa bf16 64x64 d64, in RAM (1 invocations): bitwise
      [0] cycles 373801 instret 127358 ram 0,1,2,7,8
    sdpa bf16 9x29 d64 masked (1 invocations): bitwise
      [0] cycles 44613 instret 15376 ram 2,4 |}]

let%expect_test "the check set" =
  parallel Cases.check;
  [%expect
    {|
    exp 64x64 (1 invocations): bitwise
      [0] cycles 207173 instret 99366 ram 1
    softmax over C 64x64 (1 invocations): bitwise
      [0] cycles 534046 instret 259130 ram 1
    softmax over W 16x16 (1 invocations): bitwise
      [0] cycles 70214 instret 38978
    layer norm over C 16x64 (1 invocations): bitwise
      [0] cycles 21422 instret 9362
    rms norm over C 16x64 (1 invocations): bitwise
      [0] cycles 12145 instret 5291
    linear 32x96 k48 (1 invocations): bitwise
      [0] cycles 32529 instret 12366 ram 1,3
    sdpa bf16 41x61 d32 (1 invocations): bitwise
      [0] cycles 241225 instret 84947 ram 8
    layer, bare (1 invocations): bitwise
      [0] cycles 243406 instret 162152 ram 1,2,3,4,5,7,8,9,10,11,12,13,14,15
    layer, all flags (1 invocations): bitwise
      [0] cycles 282593 instret 189982 ram 1,3,5,7,10,11,12,13,14,15,16,17,18,19,20
    layer, pre layer norm, rope (1 invocations): bitwise
      [0] cycles 387522 instret 231631 ram 3,5,7,9,12,13,14,15,16,17,18,19,20,21,22,23,24
    layer, post rms (1 invocations): bitwise
      [0] cycles 323484 instret 210827 ram 2,4,6,8,11,12,13,14,15,16,17,18,19,20,21,22
    layer decomposed, bare (14 invocations): bitwise
      [0] cycles 13922 instret 4945 ram 5,6
      [1] cycles 173009 instret 108328
      [2] cycles 2423 instret 1848
      [3] cycles 13922 instret 4945 ram 8,9
      [4] cycles 173009 instret 108328
      [5] cycles 2423 instret 1848
      [6] cycles 13922 instret 4945 ram 11,12
      [7] cycles 173009 instret 108328
      [8] cycles 2423 instret 1848
      [9] cycles 956811 instret 567243
      [10] cycles 2623 instret 2074
      [11] cycles 237237 instret 142632
      [12] cycles 13922 instret 4945 ram 17,18
      [13] cycles 2628 instret 2119 |}]

let%expect_test "the view-call set" =
  View.all ();
  Format.print_flush ();
  [%expect
    {|
    view pointwise 3 stages, 16 rows, block 2, looped (1 invocations): bitwise
      [0] cycles 3979 instret 3247
    view pointwise 3 stages, 12 rows, block 1, looped (1 invocations): bitwise
      [0] cycles 3773 instret 3223
    view pointwise 3 stages, 32 rows, block 1, looped (1 invocations): bitwise
      [0] cycles 11803 instret 9655
    view pointwise 2 stages, 32 rows, block 4, looped (1 invocations): bitwise
      [0] cycles 10352 instret 6275
    view pointwise 2 stages, 16 rows, block 2, looped (1 invocations): bitwise
      [0] cycles 8568 instret 5571
    view pointwise 1 stage, 24 rows, block 2, looped (1 invocations): bitwise
      [0] cycles 6564 instret 4611
    view window 1 stage, 3x3, 16 rows, block 2, looped (1 invocations): bitwise
      [0] cycles 15972 instret 13169
    view window 1 stage, 3x3, 24 rows, block 1, looped (1 invocations): bitwise
      [0] cycles 19500 instret 16969
    view window 1 stage, 3x3, 20 rows, block 4, looped (1 invocations): bitwise
      [0] cycles 30876 instret 23775
    view window 1 stage, 3x3, 16 rows, block 2, wide (1 invocations): bitwise
      [0] cycles 16546 instret 13043
    view pointwise 3 stages, 20 rows, block 2, wide (1 invocations): bitwise
      [0] cycles 8147 instret 5575
    view window 1 stage, 3x3, 18 rows, block 2, wide (1 invocations): bitwise
      [0] cycles 27744 instret 19171 |}]

(* A window's rows after the first add into C: with the tile started from zero
   instead of C's rows the output differs. *)
let%expect_test "a seed mutation is red on the core" =
  View.run
    ~engine_mutation:Coralnpu_kernels_zvt.Zvt_contraction.Mutation.Seed_zeroed
    (List.find
       (fun (v : Cases.View.t) ->
         v.Cases.View.name
         = "view window 1 stage, 3x3, 16 rows, block 2, looped")
       Cases.View.all);
  Format.print_flush ();
  [%expect
    {| view window 1 stage, 3x3, 16 rows, block 2, looped, mutated (1 invocations): DIFFERS |}]
