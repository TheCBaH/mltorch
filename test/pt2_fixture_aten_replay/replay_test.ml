open Pt2_fixture_unix_test.Support
module R = Pt2_fixture_replay_test.Release
module A = Pt2_fixture_aten_replay

let get r = Err.or_raise ~pp_error:U.Fixture.pp_error r

let with_fixture (release : R.t) f =
  with_dir (fun dir ->
      let transport : U.Transport.t =
       fun ~url ~dest ->
        match List.assoc_opt url release.served with
        | None -> Error "404"
        | Some data ->
            write dest data;
            Ok ()
      in
      let config = U.Bundle.config ~transport (cache dir) in
      let opened =
        get
          (U.Fixture.open_ config release.cohort
             (List.hd release.cohort.entries))
      in
      f opened)

let summarize (r : Pt2_fixture.Report.t) =
  Fmt.pr "%s %s normalizations=%d@." r.backend
    (match r.status with
    | Passed -> "passed"
    | Failed -> "failed"
    | Refused -> "refused")
    (List.length r.normalizations);
  List.iter
    (fun (c : Pt2_fixture.Report.case) ->
      Fmt.pr "%s digests=%b/%b outputs=%d mismatches=%Ld@." c.id
        c.inputs_digest_ok c.outputs_digest_ok (List.length c.outputs)
        (List.fold_left
           (fun n (o : Pt2_fixture.Compare.t) -> Int64.add n o.mismatches)
           0L c.outputs))
    r.cases

let%expect_test "three distinct typed inputs and every case/output through C++"
    =
  with_fixture (R.build ()) (fun f ->
      summarize (get (A.replay ~consumer:"test" f)));
  [%expect
    {|
    aten-cpp passed normalizations=0
    case-00 digests=true/true outputs=2 mismatches=0
    case-01 digests=true/true outputs=2 mismatches=0 |}]

let%expect_test "changed references fail the shared comparator" =
  let c = List.nth R.base_cases 1 in
  let z = R.tensor "z" R.Float (R.f32s [ 0.; -0.; 0.; 0.; -0.; 9. ]) in
  let outputs = [ List.hd c.outputs; z ] in
  let cases =
    [
      List.hd R.base_cases;
      { c with R.outputs; outputs_sha = R.content_digest outputs };
    ]
  in
  with_fixture (R.build ~cases ()) (fun f ->
      summarize (get (A.replay ~consumer:"test" f)));
  [%expect
    {|
    aten-cpp failed normalizations=0
    case-00 digests=true/true outputs=2 mismatches=0
    case-01 digests=true/true outputs=2 mismatches=1 |}]

let%expect_test "named binding refuses missing, duplicate and extra inputs" =
  with_fixture (R.build ()) (fun f ->
      let inputs =
        get
          (Pt2_archive.load_pt_tensor_map
             (Filename.concat f.bundle.dir "cases/case-00/inputs.pt"))
      in
      let run inputs =
        match Err.payload (Interp.run_named f.archive ~inputs) with
        | Error (`Input_names _) -> print_endline "input names refused"
        | Error e -> Fmt.pr "%a@." Interp.pp_error e
        | Ok _ -> print_endline "unexpected pass"
      in
      run (List.tl inputs);
      run (List.hd inputs :: inputs);
      run (("extra", snd (List.hd inputs)) :: inputs);
      match
        Err.payload (Interp.run_named f.archive ~inputs:(List.rev inputs))
      with
      | Ok outputs ->
          Fmt.pr "reordered names: %d outputs@." (List.length outputs)
      | Error e -> Fmt.pr "%a@." Interp.pp_error e);
  [%expect
    {|
    input names refused
    input names refused
    input names refused
    reordered names: 2 outputs |}]

let%expect_test "logical C++ reads honor strided and offset views, exact int64"
    =
  let ba =
    Bigarray.Array1.of_array Bigarray.int64 Bigarray.c_layout
      [| 1L; 2L; 9007199254740993L; Int64.min_int |]
  in
  let t = Aten_tensor.of_bigarray Aten_dtype.int64 ba [ 2; 2 ] in
  let offset =
    Aten_c.Aten_operations.select_int t 0L 1L
    |> Aten_tensor.check |> Aten_tensor.manage
  in
  let strided =
    Aten_c.Aten_operations.transpose_int t 0L 1L
    |> Aten_tensor.check |> Aten_tensor.manage
  in
  let show t =
    match A.to_logical t with
    | Error e -> print_endline e
    | Ok l ->
        Fmt.pr "shape [%s] [%s]@."
          (String.concat ";"
             (List.map Int64.to_string l.Pt2_fixture.Logical.shape))
          (String.concat ";"
             (List.init
                (Bigarray.Array1.dim l.data / 8)
                (fun i ->
                  Int64.to_string (Pt2_storage.get_int64_le l.data (i * 8)))))
  in
  show offset;
  show strided;
  [%expect
    {|
    shape [2] [9007199254740993;-9223372036854775808]
    shape [2;2] [1;9007199254740993;2;-9223372036854775808] |}]

let%expect_test "named binding checks actual input dtype and shape" =
  with_fixture (R.build ()) (fun f ->
      let inputs =
        get
          (Pt2_archive.load_pt_tensor_map
             (Filename.concat f.bundle.dir "cases/case-00/inputs.pt"))
      in
      let run tensor =
        let inputs =
          List.map
            (fun (name, t) -> (name, if name = "x" then tensor else t))
            inputs
        in
        match Err.payload (Interp.run_named f.archive ~inputs) with
        | Error e -> Fmt.pr "%a@." Interp.pp_error e
        | Ok _ -> print_endline "unexpected pass"
      in
      run (List.assoc "keep" inputs);
      run
        {
          (List.assoc "x" inputs) with
          Pt2_tensor.sizes = [ 6 ];
          strides = [ 1 ];
        });
  [%expect
    {|
    ATen input x metadata differs: dtype
    ATen input x metadata differs: shape |}]

let%expect_test "metadata assertions are checked against actual C++ tensors" =
  let module P = Pytorch_types in
  let module S = Schema_runtime.String_map in
  let tensor = Aten_tensor.create [ 2; 3 ] in
  let env = S.singleton "a" tensor in
  let run name arg =
    let inputs =
      [
        P.NamedArgument.make "a"
          (P.Argument.Tensor (P.TensorArgument.make "a"))
          None;
        P.NamedArgument.make name arg None;
      ]
    in
    let node =
      P.Node.make "torch.ops.aten._assert_tensor_metadata.default" inputs []
        S.empty None None
    in
    match Err.payload (Interp.assert_tensor_metadata env node) with
    | Error e -> Fmt.pr "%a@." Interp.pp_error e
    | Ok _ -> print_endline "checked"
  in
  run "size" (P.Argument.Ints [ 2; 3 ]);
  run "size" (P.Argument.Ints [ 6 ]);
  run "stride" (P.Argument.Ints [ 1; 3 ]);
  run "dtype" (P.Argument.Scalar_type P.ScalarType.FLOAT);
  run "dtype" (P.Argument.Scalar_type P.ScalarType.LONG);
  run "device" (P.Argument.Device (P.Device.make "cpu" None));
  run "device" (P.Argument.Device (P.Device.make "cuda" None));
  run "layout" (P.Argument.Layout P.Layout.Strided);
  run "layout" (P.Argument.Layout P.Layout.SparseCoo);
  [%expect
    {|
    checked
    ATen tensor metadata differs: size
    ATen tensor metadata differs: stride
    checked
    ATen tensor metadata differs: dtype
    checked
    unsupported device argument "device"
    checked
    unsupported layout argument "layout" |}]

let%expect_test "C++ diagnostics cannot enter the Native acceptance policy set"
    =
  let module P = Transformers_reports.Run.Policy in
  print_endline (P.backend P.aten);
  Fmt.pr "native policies=%d accepted=%b@." (List.length P.all)
    (Result.is_ok (Err.payload (P.of_json (P.json P.aten))));
  [%expect {|
    aten-cpp
    native policies=4 accepted=false |}]
