open Pt2_fixture_unix_test.Support
module R = Release
module Report = Pt2_fixture.Report

(* Fetch the synthetic release into a fresh cache, open it and replay. *)
let replay (rel : R.t) =
  with_dir (fun dir ->
      let calls = ref [] in
      let served = rel.served in
      let t : U.Transport.t =
       fun ~url ~dest ->
        calls := url :: !calls;
        match List.assoc_opt url served with
        | Some b ->
            write dest b;
            Ok ()
        | None -> Error "404"
      in
      let config = U.Bundle.config ~transport:t (cache dir) in
      let entry = List.hd rel.cohort.entries in
      Err.payload
        (let open Err.Syntax in
         let* f = U.Fixture.open_ config rel.cohort entry in
         Pt2_fixture_replay.replay ~consumer:"test" f))

let summarize = function
  | Error e -> show_error e
  | Ok (r : Report.t) ->
      Fmt.pr "status %s@."
        (match r.status with
        | Passed -> "passed"
        | Failed -> "failed"
        | Refused -> "refused");
      Option.iter (Fmt.pr "refusal: %s@.") r.refusal;
      List.iter
        (fun (c : Report.case) ->
          Fmt.pr "%s: inputs digest %b, outputs digest %b%s@." c.id
            c.inputs_digest_ok c.outputs_digest_ok
            (match c.error with Some e -> ", error: " ^ e | None -> "");
          List.iter
            (fun (o : Pt2_fixture.Compare.t) ->
              Fmt.pr "  %s: %d mismatches of %Ld, max abs %g%s@." o.name
                (Int64.to_int o.mismatches)
                o.elements o.max_abs_error
                (match o.first with
                | d :: _ ->
                    Printf.sprintf " first [%s] got %s want %s"
                      (String.concat ";" (List.map Int64.to_string d.index))
                      d.actual d.expected
                | [] -> ""))
            c.outputs)
        r.cases

let%expect_test "every published case passes, with digests proved" =
  summarize (replay (R.build ()));
  [%expect
    {|
    status passed
    case-00: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0 |}]

let perturb_case1_z delta =
  let case1 = List.nth R.base_cases 1 in
  let z = R.tensor "z" R.Float (R.f32s [ 0.; -0.; 0.; 0.; -0.; 8. +. delta ]) in
  [
    List.hd R.base_cases;
    {
      case1 with
      R.outputs = [ List.hd case1.outputs; z ];
      outputs_sha = R.content_digest [ List.hd case1.outputs; z ];
    };
  ]

let%expect_test "the producer's tolerance decides, and nothing is relaxed" =
  (* 8 + d against 8: allowed error is 1e-5 + 1e-4 * 8 = 8.1e-4 *)
  print_endline "-- within tolerance";
  summarize (replay (R.build ~cases:(perturb_case1_z 5e-4) ()));
  print_endline "-- just outside";
  summarize (replay (R.build ~cases:(perturb_case1_z 1e-3) ()));
  print_endline "-- a tighter contract tolerance, same data";
  summarize
    (replay
       (let tight_contract =
          R.contract_json ~atol:"1e-08" ~rtol:"1e-08" (R.program ())
        in
        let tight_cases =
          R.cases_json ~atol:"1e-08" ~rtol:"1e-08" (perturb_case1_z 5e-4)
        in
        R.build ~contract:tight_contract ~cases_text:tight_cases
          ~cases:(perturb_case1_z 5e-4) ()));
  [%expect
    {|
    -- within tolerance
    status passed
    case-00: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0.000499725
    -- just outside
    status failed
    case-00: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 1 mismatches of 6, max abs 0.0010004 first [1;2] got 8 want 8.0010004
    -- a tighter contract tolerance, same data
    status failed
    case-00: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 1 mismatches of 6, max abs 0.000499725 first [1;2] got 8 want 8.00049973 |}]

let%expect_test
    "wrong values, swapped outputs and a lying descriptor are all caught" =
  let case0 = List.hd R.base_cases and case1 = List.nth R.base_cases 1 in
  print_endline "-- the reference y is wrong in one element";
  let y_bad = R.tensor "y" R.Float (R.f32s [ 11.; 22.; 33.; 44.; 55.; 67. ]) in
  let bad = { case0 with R.outputs = [ y_bad; List.nth case0.outputs 1 ] } in
  summarize
    (replay
       (R.build
          ~cases:
            [ { bad with R.outputs_sha = R.content_digest bad.outputs }; case1 ]
          ()));
  print_endline "-- y and z swapped in outputs.pt (names kept, data exchanged)";
  let swapped =
    [
      R.tensor "y" R.Float (List.nth case0.outputs 1).R.raw;
      R.tensor "z" R.Float (List.hd case0.outputs).R.raw;
    ]
  in
  summarize
    (replay
       (R.build
          ~cases:
            [
              {
                case0 with
                R.outputs = swapped;
                outputs_sha = R.content_digest swapped;
              };
              case1;
            ]
          ()));
  print_endline "-- outputs.pt matches nothing the descriptor says";
  summarize
    (replay (R.build ~cases:[ { case0 with R.outputs = swapped }; case1 ] ()));
  print_endline
    "-- the inputs file was altered after the descriptor was written";
  let x_other = R.tensor "x" R.Float (R.f32s [ 9.; 9.; 9.; 9.; 9.; 9. ]) in
  summarize
    (replay
       (R.build
          ~cases:
            [ { case0 with R.inputs = x_other :: List.tl case0.inputs }; case1 ]
          ()));
  [%expect
    {|
    -- the reference y is wrong in one element
    status failed
    case-00: inputs digest true, outputs digest true
      y: 1 mismatches of 6, max abs 1 first [1;2] got 66 want 67
      z: 0 mismatches of 6, max abs 0
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    -- y and z swapped in outputs.pt (names kept, data exchanged)
    status failed
    case-00: inputs digest true, outputs digest true
      y: 6 mismatches of 6, max abs 66 first [0;0] got 11 want 1
      z: 6 mismatches of 6, max abs 66 first [0;0] got 1 want 11
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    -- outputs.pt matches nothing the descriptor says
    status failed
    case-00: inputs digest true, outputs digest false
      y: 6 mismatches of 6, max abs 66 first [0;0] got 11 want 1
      z: 6 mismatches of 6, max abs 66 first [0;0] got 1 want 11
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    -- the inputs file was altered after the descriptor was written
    status failed
    case-00: inputs digest false, outputs digest true
      y: 6 mismatches of 6, max abs 8 first [0;0] got 19 want 11
      z: 3 mismatches of 6, max abs 8 first [0;0] got 9 want 1
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0 |}]

let%expect_test "structural faults: names, dtypes, counts" =
  let case0 = List.hd R.base_cases and case1 = List.nth R.base_cases 1 in
  print_endline "-- an input tensor of the wrong dtype";
  let ids_f =
    R.tensor "ids" R.Float (R.f32s [ 10.; 20.; 30.; 40.; 50.; 60. ])
  in
  let wrong = [ List.hd case0.inputs; ids_f; List.nth case0.inputs 2 ] in
  summarize
    (replay
       (R.build
          ~cases:
            [
              {
                case0 with
                R.inputs = wrong;
                inputs_sha = R.content_digest wrong;
              };
              case1;
            ]
          ()));
  print_endline "-- inputs.pt holds a differently named set";
  let renamed =
    [
      List.hd case0.inputs;
      List.nth case0.inputs 1;
      { (List.nth case0.inputs 2) with R.name = "mask" };
    ]
  in
  summarize
    (replay (R.build ~cases:[ { case0 with R.inputs = renamed }; case1 ] ()));
  print_endline "-- tolerances disagree between cases.json and the contract";
  summarize
    (replay (R.build ~cases_text:(R.cases_json ~atol:"1e-06" R.base_cases) ()));
  print_endline "-- keyword order differs from the input list";
  summarize
    (replay
       (R.build
          ~contract:
            (R.contract_json ~kwargs:{|["ids","x","keep"]|} (R.program ()))
          ()));
  [%expect
    {|
    -- an input tensor of the wrong dtype
    status failed
    case-00: inputs digest true, outputs digest true, error: input tensors differ from the contract: ids
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    -- inputs.pt holds a differently named set
    status failed
    case-00: inputs digest false, outputs digest false, error: inputs.pt holds [ids; mask; x], expected [x; ids; keep]
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    -- tolerances disagree between cases.json and the contract
    cases.json: tolerances is "atol=1e-06 rtol=0.0001", expected "atol=1e-05 rtol=0.0001"
    -- keyword order differs from the input list
    the contract is not a plain tensor call: keyword order differs from the input list |}]

(* An operator the engine does not implement is a refusal of the artifact. *)
let%expect_test "a graph the engine cannot run is refused, not passed" =
  summarize
    (replay (R.build ~program:(R.program ~op:"floor_divide.default" ()) ()));
  [%expect
    {|
    status refused
    refusal: unsupported PT2 operator: torch.ops.aten.floor_divide.default
    case-00: inputs digest true, outputs digest true, error: unsupported PT2 operator: torch.ops.aten.floor_divide.default
    case-01: inputs digest true, outputs digest true, error: unsupported PT2 operator: torch.ops.aten.floor_divide.default |}]

(* Right values under a descriptor that does not describe them: the tensors
   match the reference elementwise, but the case is not what it claims to be. *)
let%expect_test "a case that fails its own digest never passes" =
  let case0 = List.hd R.base_cases and case1 = List.nth R.base_cases 1 in
  print_endline "-- inputs digest wrong, values right";
  summarize
    (replay
       (R.build
          ~cases:[ { case0 with R.inputs_sha = String.make 64 '0' }; case1 ]
          ()));
  print_endline "-- outputs digest wrong, values right";
  summarize
    (replay
       (R.build
          ~cases:[ case0; { case1 with R.outputs_sha = String.make 64 '1' } ]
          ()));
  [%expect
    {|
    -- inputs digest wrong, values right
    status failed
    case-00: inputs digest false, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    case-01: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    -- outputs digest wrong, values right
    status failed
    case-00: inputs digest true, outputs digest true
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0
    case-01: inputs digest true, outputs digest false
      y: 0 mismatches of 6, max abs 0
      z: 0 mismatches of 6, max abs 0 |}]
