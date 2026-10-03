open Loop_ir
open Loop_fixtures
open Loop_programs

(* Execution marks have a testable counting path: a counting build bumps one word
   per mark, and the counts must equal the interpreter's. The default build
   emits nothing for a mark, so inference never pays for them or calls out. *)

let data = [| 1.; 2.; 3.; 10.; 20.; 30. |]

let bind id =
  if Tensor_id.equal id (tid 0) then
    Some
      (f32_tensor rows_shape (fun c -> data.((Vec6.offset rows_shape c :> int))))
  else None

let program_of kernel =
  match Loop_lower.lower (Fusion_plan.default kernel) with
  | Ok p -> p
  | Error _ -> failwith "lowering refused the kernel"

let interp_counts p =
  let c = Loop_interp.counters () in
  ignore (run_ok ~counters:c p ~bind);
  [
    (Loop_mark.Emitter, c.Loop_interp.emitters);
    (Loop_mark.Key, c.Loop_interp.keys);
    (Loop_mark.Local, c.Loop_interp.locals);
    (Loop_mark.Reduction, c.Loop_interp.reductions);
    (Loop_mark.Scan, c.Loop_interp.scans);
    (Loop_mark.Scan_update, c.Loop_interp.scan_updates);
  ]

let show name kernel =
  let p = program_of kernel in
  let wasm =
    match Err.payload (Loop_wasm_exec.exec_counted p ~bind) with
    | Ok (_, counts) -> counts
    | Error e -> Fmt.failwith "%s: %a" name Loop_wasm_exec.pp_error e
  in
  let interp = interp_counts p in
  Fmt.pr "%s: %s; counts match: %b@." name
    (String.concat " "
       (List.map
          (fun (m, n) -> Printf.sprintf "%s=%d" (Loop_mark.name m) n)
          wasm))
    (wasm = interp)

let%expect_test "marks: the counting build agrees with the interpreter" =
  show "centered" (region_kernel_of centered_program);
  show "extent 1" (region_kernel_of (vector_program ~extent:1 ~pick:(pick 0)));
  show "extent 3" (region_kernel_of (vector_program ~extent:3 ~pick:(pick 2)));
  show "whole only" (region_kernel_of whole_only_program);
  show "trace scan" (region_kernel_of (trace_program ~steps:2));
  show "inline scan" (inline_scan_kernel ~steps:2);
  [%expect
    {|
    centered: emitter=6 key=2 local=2 reduction=6 scan=0 scan_update=0; counts match: true
    extent 1: emitter=6 key=2 local=2 reduction=0 scan=0 scan_update=0; counts match: true
    extent 3: emitter=6 key=2 local=6 reduction=0 scan=0 scan_update=0; counts match: true
    whole only: emitter=6 key=1 local=1 reduction=0 scan=0 scan_update=0; counts match: true
    trace scan: emitter=6 key=2 local=18 reduction=0 scan=2 scan_update=12; counts match: true
    inline scan: emitter=0 key=0 local=0 reduction=0 scan=0 scan_update=0; counts match: true |}]

let%expect_test "the default build emits nothing for a mark" =
  let p = program_of (region_kernel_of centered_program) in
  let lowered build =
    match Err.payload build with
    | Ok l -> l
    | Error e -> Fmt.failwith "%a" Loop_wasm.pp_error e
  in
  let plain = lowered (Loop_wasm.lower p) in
  let counting = lowered (Loop_wasm.lower ~count_marks:true p) in
  let size l =
    String.length (Result.get_ok (Err.payload (Loop_wasm.encode l)))
  in
  Fmt.pr "mark base: %a / %a@."
    Fmt.(option ~none:(any "none") int)
    plain.Loop_wasm.mark_base
    Fmt.(option ~none:(any "none") int)
    counting.Loop_wasm.mark_base;
  Fmt.pr "counting build is larger: %b@." (size counting > size plain);
  [%expect {|
    mark base: none / 112
    counting build is larger: true |}]
