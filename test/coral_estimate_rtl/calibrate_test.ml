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
      [0] cycles 23404 instret 10991
    cos 16x16, scalar (1 invocations): bitwise
      [0] cycles 72851 instret 40722
    cos 32x32 (1 invocations): bitwise
      [0] cycles 92716 instret 43311
    cos 32x32, scalar (1 invocations): bitwise
      [0] cycles 290764 instret 162235
    exp 16x16 (1 invocations): bitwise
      [0] cycles 38028 instret 17441
    exp 16x16, scalar (1 invocations): bitwise
      [0] cycles 118121 instret 56057
    exp 32x32 (1 invocations): bitwise
      [0] cycles 151212 instret 69105
    exp 32x32, scalar (1 invocations): bitwise
      [0] cycles 471801 instret 223561
    log 16x16 (1 invocations): bitwise
      [0] cycles 28187 instret 13884
    log 16x16, scalar (1 invocations): bitwise
      [0] cycles 108809 instret 54121
    log 32x32 (1 invocations): bitwise
      [0] cycles 111851 instret 54892
    log 32x32, scalar (1 invocations): bitwise
      [0] cycles 434588 instret 215825
    sin 16x16 (1 invocations): bitwise
      [0] cycles 23052 instret 11041
    sin 16x16, scalar (1 invocations): bitwise
      [0] cycles 71747 instret 40354
    sin 32x32 (1 invocations): bitwise
      [0] cycles 91308 instret 43505
    sin 32x32, scalar (1 invocations): bitwise
      [0] cycles 286335 instret 160756
    sqrt 16x16 (1 invocations): bitwise
      [0] cycles 1563 instret 751
    sqrt 16x16, scalar (1 invocations): bitwise
      [0] cycles 8093 instret 2607
    sqrt 32x32 (1 invocations): bitwise
      [0] cycles 5387 instret 2351
    sqrt 32x32, scalar (1 invocations): bitwise
      [0] cycles 31695 instret 9775
    add 16x64 (1 invocations): bitwise
      [0] cycles 3127 instret 2667
    add 16x64, scalar (1 invocations): bitwise
      [0] cycles 18765 instret 11619
    add 32x128 (1 invocations): bitwise
      [0] cycles 18711 instret 9883 ram 1,2
    add 32x128, scalar (1 invocations): bitwise
      [0] cycles 86637 instret 45715 ram 1,2
    div 16x64 (1 invocations): bitwise
      [0] cycles 5274 instret 2667
    div 16x64, scalar (1 invocations): bitwise
      [0] cycles 30936 instret 11619
    div 32x128 (1 invocations): bitwise
      [0] cycles 21672 instret 9883 ram 1,2
    div 32x128, scalar (1 invocations): bitwise
      [0] cycles 135333 instret 45715 ram 1,2
    mul 16x64 (1 invocations): bitwise
      [0] cycles 3127 instret 2667
    mul 16x64, scalar (1 invocations): bitwise
      [0] cycles 18765 instret 11619
    mul 32x128 (1 invocations): bitwise
      [0] cycles 18711 instret 9883 ram 1,2
    mul 32x128, scalar (1 invocations): bitwise
      [0] cycles 86637 instret 45715 ram 1,2
    x*2 16x64 (1 invocations): bitwise
      [0] cycles 2423 instret 2107
    x*2 16x64, scalar (1 invocations): bitwise
      [0] cycles 15658 instret 9519
    x*2 32x128 (1 invocations): bitwise
      [0] cycles 12761 instret 7739 ram 1
    x*2 32x128, scalar (1 invocations): bitwise
      [0] cycles 61963 instret 37423 ram 1
    x*x+x 16x64 (2 invocations): bitwise
      [0] cycles 3061 instret 2357
      [1] cycles 3127 instret 2667
    x*x+x 16x64, scalar (2 invocations): bitwise
      [0] cycles 15657 instret 9520
      [1] cycles 18765 instret 11619
    x*x+x 32x128 (2 invocations): bitwise
      [0] cycles 15447 instret 8757 ram 1
      [1] cycles 18711 instret 9883 ram 0,2
    x*x+x 32x128, scalar (2 invocations): bitwise
      [0] cycles 61962 instret 37424 ram 1
      [1] cycles 86637 instret 45715 ram 0,2
    x+y+z 16x64 (2 invocations): bitwise
      [0] cycles 3127 instret 2667
      [1] cycles 3127 instret 2667
    x+y+z 16x64, scalar (2 invocations): bitwise
      [0] cycles 18765 instret 11619
      [1] cycles 18765 instret 11619
    x+y+z 32x128 (2 invocations): bitwise
      [0] cycles 18711 instret 9883 ram 1,3
      [1] cycles 18711 instret 9883 ram 2,4
    x+y+z 32x128, scalar (2 invocations): bitwise
      [0] cycles 86637 instret 45715 ram 1,3
      [1] cycles 86637 instret 45715 ram 2,4
    add 16x64, in RAM (1 invocations): bitwise
      [0] cycles 5671 instret 2667 ram 0,1,2
    add 32x128, in RAM (1 invocations): bitwise
      [0] cycles 21751 instret 9883 ram 0,1,2
    add 16x64, scalar, in RAM (1 invocations): bitwise
      [0] cycles 24925 instret 11619 ram 0,1,2
    softmax over C 16x16 (1 invocations): bitwise
      [0] cycles 262426 instret 125468
    softmax over C 32x32 (1 invocations): bitwise
      [0] cycles 1048224 instret 500376
    sum over C 16x64 (1 invocations): bitwise
      [0] cycles 1838 instret 1747
    sum over C 32x128 (1 invocations): bitwise
      [0] cycles 7055 instret 6772
    sum over W 16x64 (1 invocations): bitwise
      [0] cycles 1544 instret 1462
    transpose 32x32 (1 invocations): bitwise
      [0] cycles 3403 instret 2798
    transpose 64x64 (1 invocations): bitwise
      [0] cycles 15947 instret 10030 ram 1
    transpose 32x32, in RAM (1 invocations): bitwise
      [0] cycles 7499 instret 2798 ram 0,1
    head split 16x64, 2 heads (2 invocations): bitwise
      [0] cycles 173202 instret 108489
      [1] cycles 2598 instret 2186
    head split 32x128, 4 heads (2 invocations): bitwise
      [0] cycles 732355 instret 433003 ram 1
      [1] cycles 14113 instret 8453 ram 2
    head split 49x64, 2 heads (2 invocations): bitwise
      [0] cycles 567877 instret 338369 ram 1
      [1] cycles 10816 instret 6476 ram 2
    linear 9x32 k64 (1 invocations): bitwise
      [0] cycles 4895 instret 2240
    linear 16x64 k64 (1 invocations): bitwise
      [0] cycles 14824 instret 5222 ram 2,4
    linear 64x64 k64 (1 invocations): bitwise
      [0] cycles 107982 instret 20624 ram 1,2,4
    linear 16x128 k32 (1 invocations): bitwise
      [0] cycles 20036 instret 7286 ram 2,4
    linear 32x32 k128 (1 invocations): bitwise
      [0] cycles 47891 instret 8295 ram 1,4
    linear 48x80 k32 (1 invocations): bitwise
      [0] cycles 37714 instret 13892 ram 2,4
    linear bf16 9x32 k64 (1 invocations): bitwise
      [0] cycles 4241 instret 1895
    linear bf16 16x64 k64 (1 invocations): bitwise
      [0] cycles 11304 instret 4242 ram 2,4
    linear bf16 64x64 k64 (1 invocations): bitwise
      [0] cycles 37789 instret 12752 ram 1,2,4
    linear bf16 32x128 k32 (1 invocations): bitwise
      [0] cycles 27091 instret 10940 ram 2,4
    linear bf16 48x48 k48 (1 invocations): bitwise
      [0] cycles 18544 instret 7863 ram 2,4
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
      [0] cycles 15217 instret 2240 ram 0,1,2,3,4
    linear 32x32 k64, in RAM (1 invocations): bitwise
      [0] cycles 39637 instret 5479 ram 0,1,2,3,4
    linear bf16 16x256 k64 (1 invocations): bitwise
      [0] cycles 103194 instret 16334 ram 1,4
    linear bf16 32x128 k128 (1 invocations): bitwise
      [0] cycles 193518 instret 24266 ram 1,2,4
    linear bf16 49x512 k64 (1 invocations): bitwise
      [0] cycles 787031 instret 117164 ram 1,2,4
    linear bf16 16x256 k64, in RAM (1 invocations): bitwise
      [0] cycles 156412 instret 16334 ram 0,1,2,3,4
    linear bf16 16x64 k64, in RAM (1 invocations): bitwise
      [0] cycles 18984 instret 4242 ram 0,1,2,3,4
    linear bf16 64x64 k64, in RAM (1 invocations): bitwise
      [0] cycles 43933 instret 12752 ram 0,1,2,3,4
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
      [0] cycles 607246 instret 275249 ram 1
    softmax over C 64x64 (1 invocations): bitwise
      [0] cycles 4190117 instret 1998734 ram 1
    softmax over W 16x16 (1 invocations): bitwise
      [0] cycles 262426 instret 125468
    layer norm over C 16x64 (1 invocations): bitwise
      [0] cycles 148915 instret 83187
    rms norm over C 16x64 (1 invocations): bitwise
      [0] cycles 61083 instret 34604
    linear 32x96 k48 (1 invocations): bitwise
      [0] cycles 61656 instret 13139 ram 1,4
    sdpa bf16 41x61 d32 (1 invocations): bitwise
      [0] cycles 241222 instret 84947 ram 8
    layer, bare (1 invocations): bitwise
      [0] cycles 260314 instret 121585 ram 1,2,3,4,5,7,8,9,10,11,12,13,14,15
    layer, all flags (1 invocations): bitwise
      [0] cycles 323334 instret 144122 ram 1,3,5,7,10,11,12,13,14,15,16,17,18,19,20
    layer, pre layer norm, rope (1 invocations): bitwise
      [0] cycles 436619 instret 195124 ram 3,5,7,9,12,13,14,15,16,17,18,19,20,21,22,23,24
    layer, post rms (1 invocations): bitwise
      [0] cycles 373465 instret 167100 ram 2,4,6,8,11,12,13,14,15,16,17,18,19,20,21,22
    layer decomposed, bare (14 invocations): bitwise
      [0] cycles 14824 instret 5222 ram 5,20
      [1] cycles 173202 instret 108489
      [2] cycles 2598 instret 2186
      [3] cycles 14824 instret 5222 ram 8,20
      [4] cycles 173202 instret 108489
      [5] cycles 2598 instret 2186
      [6] cycles 14824 instret 5222 ram 11,20
      [7] cycles 173202 instret 108489
      [8] cycles 2598 instret 2186
      [9] cycles 1982524 instret 1233524
      [10] cycles 2911 instret 2511
      [11] cycles 237302 instret 142681
      [12] cycles 14824 instret 5222 ram 17,20
      [13] cycles 3127 instret 2667 |}]
