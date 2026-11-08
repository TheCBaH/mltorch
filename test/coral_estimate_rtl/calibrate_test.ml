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
      [0] cycles 1235 instret 486
    sqrt 16x16, scalar (1 invocations): bitwise
      [0] cycles 8035 instret 2552
    sqrt 32x32 (1 invocations): bitwise
      [0] cycles 4739 instret 1830
    sqrt 32x32, scalar (1 invocations): bitwise
      [0] cycles 31589 instret 9672
    add 16x64 (1 invocations): bitwise
      [0] cycles 2629 instret 2119
    add 16x64, scalar (1 invocations): bitwise
      [0] cycles 18690 instret 11562
    add 32x128 (1 invocations): bitwise
      [0] cycles 17477 instret 8359 ram 1,2
    add 32x128, scalar (1 invocations): bitwise
      [0] cycles 86498 instret 45610 ram 1,2
    div 16x64 (1 invocations): bitwise
      [0] cycles 4908 instret 2343
    div 16x64, scalar (1 invocations): bitwise
      [0] cycles 30861 instret 11562
    div 32x128 (1 invocations): bitwise
      [0] cycles 20950 instret 9255 ram 1,2
    div 32x128, scalar (1 invocations): bitwise
      [0] cycles 135194 instret 45610 ram 1,2
    mul 16x64 (1 invocations): bitwise
      [0] cycles 2629 instret 2119
    mul 16x64, scalar (1 invocations): bitwise
      [0] cycles 18690 instret 11562
    mul 32x128 (1 invocations): bitwise
      [0] cycles 17477 instret 8359 ram 1,2
    mul 32x128, scalar (1 invocations): bitwise
      [0] cycles 86498 instret 45610 ram 1,2
    x*2 16x64 (1 invocations): bitwise
      [0] cycles 1899 instret 1613
    x*2 16x64, scalar (1 invocations): bitwise
      [0] cycles 15592 instret 9470
    x*2 32x128 (1 invocations): bitwise
      [0] cycles 11340 instret 6317 ram 1
    x*2 32x128, scalar (1 invocations): bitwise
      [0] cycles 61833 instret 37326 ram 1
    x*x+x 16x64 (2 invocations): bitwise
      [0] cycles 2626 instret 1862
      [1] cycles 2629 instret 2119
    x*x+x 16x64, scalar (2 invocations): bitwise
      [0] cycles 15584 instret 9465
      [1] cycles 18690 instret 11562
    x*x+x 32x128 (2 invocations): bitwise
      [0] cycles 14402 instret 7334 ram 1
      [1] cycles 17477 instret 8359 ram 0,2
    x*x+x 32x128, scalar (2 invocations): bitwise
      [0] cycles 61825 instret 37321 ram 1
      [1] cycles 86498 instret 45610 ram 0,2
    x+y+z 16x64 (2 invocations): bitwise
      [0] cycles 2629 instret 2119
      [1] cycles 2629 instret 2119
    x+y+z 16x64, scalar (2 invocations): bitwise
      [0] cycles 18690 instret 11562
      [1] cycles 18690 instret 11562
    x+y+z 32x128 (2 invocations): bitwise
      [0] cycles 17477 instret 8359 ram 1,3
      [1] cycles 17477 instret 8359 ram 2,4
    x+y+z 32x128, scalar (2 invocations): bitwise
      [0] cycles 86498 instret 45610 ram 1,3
      [1] cycles 86498 instret 45610 ram 2,4
    add 16x64, in RAM (1 invocations): bitwise
      [0] cycles 5188 instret 2119 ram 0,1,2
    add 32x128, in RAM (1 invocations): bitwise
      [0] cycles 20548 instret 8359 ram 0,1,2
    div 16x64, in RAM (1 invocations): bitwise
      [0] cycles 6055 instret 2343 ram 0,1,2
    div 32x128, in RAM (1 invocations): bitwise
      [0] cycles 24021 instret 9255 ram 0,1,2
    exp 32x32, in RAM (1 invocations): bitwise
      [0] cycles 51971 instret 24870 ram 0,1
    add 16x64, scalar, in RAM (1 invocations): bitwise
      [0] cycles 24850 instret 11562 ram 0,1,2
    softmax over C 16x16 (1 invocations): bitwise
      [0] cycles 34940 instret 17333
    softmax over C 32x32 (1 invocations): bitwise
      [0] cycles 135948 instret 66613
    sum over C 16x64 (1 invocations): bitwise
      [0] cycles 1826 instret 1741
    sum over C 32x128 (1 invocations): bitwise
      [0] cycles 7036 instret 6762
    sum over W 16x64 (1 invocations): bitwise
      [0] cycles 1514 instret 1440
    transpose 32x32 (1 invocations): bitwise
      [0] cycles 3172 instret 2252
    transpose 64x64 (1 invocations): bitwise
      [0] cycles 15492 instret 8108 ram 1
    transpose 32x32, in RAM (1 invocations): bitwise
      [0] cycles 7268 instret 2252 ram 0,1
    head split 16x64, 2 heads (2 invocations): bitwise
      [0] cycles 173009 instret 108328
      [1] cycles 2455 instret 1848
    head split 32x128, 4 heads (2 invocations): bitwise
      [0] cycles 731737 instret 432548 ram 1
      [1] cycles 13578 instret 7151 ram 2
    head split 49x64, 2 heads (2 invocations): bitwise
      [0] cycles 567288 instret 337878 ram 1
      [1] cycles 10416 instret 5481 ram 2
    linear 9x32 k64 (1 invocations): bitwise
      [0] cycles 4844 instret 2145
    linear 16x64 k64 (1 invocations): bitwise
      [0] cycles 14754 instret 4945 ram 2,4
    linear 64x64 k64 (1 invocations): bitwise
      [0] cycles 107721 instret 19534 ram 1,2,4
    linear 16x128 k32 (1 invocations): bitwise
      [0] cycles 19966 instret 6785 ram 2,4
    linear 32x32 k128 (1 invocations): bitwise
      [0] cycles 47758 instret 7973 ram 1,4
    linear 48x80 k32 (1 invocations): bitwise
      [0] cycles 37517 instret 13360 ram 2,4
    linear bf16 9x32 k64 (1 invocations): bitwise
      [0] cycles 4192 instret 1800
    linear bf16 16x64 k64 (1 invocations): bitwise
      [0] cycles 11236 instret 3965 ram 2,4
    linear bf16 64x64 k64 (1 invocations): bitwise
      [0] cycles 37530 instret 11662 ram 1,2,4
    linear bf16 32x128 k32 (1 invocations): bitwise
      [0] cycles 26959 instret 9943 ram 2,4
    linear bf16 48x48 k48 (1 invocations): bitwise
      [0] cycles 18355 instret 7673 ram 2,4
    sdpa bf16 32x32 d32 (1 invocations): bitwise
      [0] cycles 87669 instret 30256
    sdpa bf16 64x64 d64 (1 invocations): bitwise
      [0] cycles 386501 instret 127358 ram 0,1,2,8
    sdpa bf16 16x48 d32 (1 invocations): bitwise
      [0] cycles 66170 instret 23479
    sdpa bf16 16x16 d32 bk16 (1 invocations): bitwise
      [0] cycles 25119 instret 8690
    sdpa bf16 32x64 d64 bk16 (1 invocations): bitwise
      [0] cycles 208447 instret 72454 ram 1,2,8
    sdpa bf16 64x32 d32 (1 invocations): bitwise
      [0] cycles 175321 instret 60394 ram 8
    linear 9x32 k64, in RAM (1 invocations): bitwise
      [0] cycles 15175 instret 2145 ram 0,1,2,3,4
    linear 32x32 k64, in RAM (1 invocations): bitwise
      [0] cycles 39504 instret 5157 ram 0,1,2,3,4
    linear bf16 16x256 k64 (1 invocations): bitwise
      [0] cycles 103124 instret 15385 ram 1,4
    linear bf16 32x128 k128 (1 invocations): bitwise
      [0] cycles 193384 instret 23269 ram 1,2,4
    linear bf16 49x512 k64 (1 invocations): bitwise
      [0] cycles 788876 instret 111523 ram 1,2,4
    linear bf16 16x256 k64, in RAM (1 invocations): bitwise
      [0] cycles 156342 instret 15385 ram 0,1,2,3,4
    linear bf16 16x64 k64, in RAM (1 invocations): bitwise
      [0] cycles 18916 instret 3965 ram 0,1,2,3,4
    linear bf16 64x64 k64, in RAM (1 invocations): bitwise
      [0] cycles 43674 instret 11662 ram 0,1,2,3,4
    sdpa bf16 32x32 d32, in RAM (1 invocations): bitwise
      [0] cycles 89717 instret 30256 ram 0,1,2,7,8
    sdpa bf16 16x16 d32 bk16, in RAM (1 invocations): bitwise
      [0] cycles 25759 instret 8690 ram 0,1,2,7,8
    sdpa bf16 64x64 d64, in RAM (1 invocations): bitwise
      [0] cycles 386501 instret 127358 ram 0,1,2,7,8
    sdpa bf16 9x29 d64 masked (1 invocations): bitwise
      [0] cycles 45247 instret 15376 ram 2,4 |}]

let%expect_test "the check set" =
  parallel Cases.check;
  [%expect
    {|
    exp 64x64 (1 invocations): bitwise
      [0] cycles 207173 instret 99366 ram 1
    softmax over C 64x64 (1 invocations): bitwise
      [0] cycles 536655 instret 261173 ram 1
    softmax over W 16x16 (1 invocations): bitwise
      [0] cycles 70886 instret 39491
    layer norm over C 16x64 (1 invocations): bitwise
      [0] cycles 22678 instret 10345
    rms norm over C 16x64 (1 invocations): bitwise
      [0] cycles 12627 instret 5639
    linear 32x96 k48 (1 invocations): bitwise
      [0] cycles 61522 instret 12366 ram 1,4
    sdpa bf16 41x61 d32 (1 invocations): bitwise
      [0] cycles 241222 instret 84947 ram 8
    layer, bare (1 invocations): bitwise
      [0] cycles 277342 instret 162152 ram 1,2,3,4,5,7,8,9,10,11,12,13,14,15
    layer, all flags (1 invocations): bitwise
      [0] cycles 320945 instret 189982 ram 1,3,5,7,10,11,12,13,14,15,16,17,18,19,20
    layer, pre layer norm, rope (1 invocations): bitwise
      [0] cycles 429970 instret 231631 ram 3,5,7,9,12,13,14,15,16,17,18,19,20,21,22,23,24
    layer, post rms (1 invocations): bitwise
      [0] cycles 363868 instret 210827 ram 2,4,6,8,11,12,13,14,15,16,17,18,19,20,21,22
    layer decomposed, bare (14 invocations): bitwise
      [0] cycles 14754 instret 4945 ram 5,20
      [1] cycles 173009 instret 108328
      [2] cycles 2455 instret 1848
      [3] cycles 14754 instret 4945 ram 8,20
      [4] cycles 173009 instret 108328
      [5] cycles 2455 instret 1848
      [6] cycles 14754 instret 4945 ram 11,20
      [7] cycles 173009 instret 108328
      [8] cycles 2455 instret 1848
      [9] cycles 1147441 instret 717421
      [10] cycles 2655 instret 2074
      [11] cycles 237237 instret 142632
      [12] cycles 14754 instret 4945 ram 17,20
      [13] cycles 2629 instret 2119 |}]
