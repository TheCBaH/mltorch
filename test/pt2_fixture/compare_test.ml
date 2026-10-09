module F = Pt2_fixture
module Dtype = Pt2_checkpoint_map.Dtype

let f32 shape values =
  let b = Bytes.create (4 * List.length values) in
  List.iteri
    (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v))
    values;
  match
    Err.payload (F.Logical.of_bytes ~dtype:Dtype.F32 ~shape (Bytes.to_string b))
  with
  | Ok l -> l
  | Error _ -> failwith "f32"

let i64 shape values =
  let b = Bytes.create (8 * List.length values) in
  List.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) values;
  match
    Err.payload (F.Logical.of_bytes ~dtype:Dtype.I64 ~shape (Bytes.to_string b))
  with
  | Ok l -> l
  | Error _ -> failwith "i64"

let bool shape values =
  let s =
    String.concat "" (List.map (fun v -> if v then "\001" else "\000") values)
  in
  match Err.payload (F.Logical.of_bytes ~dtype:Dtype.BOOL ~shape s) with
  | Ok l -> l
  | Error _ -> failwith "bool"

let show (c : F.Compare.t) =
  let verdict =
    match c.verdict with
    | F.Compare.Pass -> "pass"
    | Values_differ -> "values differ"
    | Dtype_differs { actual; expected } ->
        Fmt.str "dtype %a vs %a" Dtype.pp actual Dtype.pp expected
    | Shape_differs _ -> "shape differs"
    | Unsupported_dtype d -> Fmt.str "unsupported %a" Dtype.pp d
  in
  Fmt.pr "%s: %s, %Ld/%Ld mismatches%s@." c.name verdict c.mismatches c.elements
    (match c.first with
    | [] -> ""
    | ds ->
        " ["
        ^ String.concat "; "
            (List.map
               (fun (d : F.Compare.Diff.t) ->
                 Printf.sprintf "%s:%s/%s"
                   (String.concat "," (List.map Int64.to_string d.index))
                   d.actual d.expected)
               ds)
        ^ "]")

let cmp ?(atol = 1e-5) ?(rtol = 1e-4) name expected actual =
  show (F.Compare.tensor ~atol ~rtol ~name ~expected ~actual)

let%expect_test "float elements: equal, close, far, special values" =
  let e = f32 [ 6L ] [ 1.; 1000.; 0.; infinity; neg_infinity; 5. ] in
  cmp "identical" e e;
  (* 1000 allows 1e-5 + 0.1 = 0.10001; 1.1 vs 1: allows 1.1e-4 *)
  cmp "within" e
    (f32 [ 6L ] [ 1.00005; 1000.09; 0.; infinity; neg_infinity; 5. ]);
  cmp "outside" e (f32 [ 6L ] [ 1.001; 1000.2; 0.; infinity; neg_infinity; 5. ]);
  cmp "inf sign" e
    (f32 [ 6L ] [ 1.; 1000.; 0.; neg_infinity; neg_infinity; 5. ]);
  cmp "finite for inf" e
    (f32 [ 6L ] [ 1.; 1000.; 0.; 3.4e38; neg_infinity; 5. ]);
  cmp "signed zero" (f32 [ 2L ] [ 0.; -0. ]) (f32 [ 2L ] [ -0.; 0. ]);
  (* NaN never matches, even against NaN: equal_nan is false. *)
  cmp "nan vs nan" (f32 [ 1L ] [ nan ]) (f32 [ 1L ] [ nan ]);
  cmp "nan vs 0" (f32 [ 1L ] [ 0. ]) (f32 [ 1L ] [ nan ]);
  cmp "tiny against zero" (f32 [ 1L ] [ 0. ]) (f32 [ 1L ] [ 9e-6 ]);
  cmp "just over atol against zero" (f32 [ 1L ] [ 0. ]) (f32 [ 1L ] [ 1.1e-5 ]);
  [%expect
    {|
    identical: pass, 0/6 mismatches
    within: pass, 0/6 mismatches
    outside: values differ, 2/6 mismatches [0:1.00100005/1; 1:1000.20001/1000]
    inf sign: values differ, 1/6 mismatches [3:-inf/inf]
    finite for inf: values differ, 1/6 mismatches [3:3.39999995e+38/inf]
    signed zero: pass, 0/2 mismatches
    nan vs nan: values differ, 1/1 mismatches [0:nan/nan]
    nan vs 0: values differ, 1/1 mismatches [0:nan/0]
    tiny against zero: pass, 0/1 mismatches
    just over atol against zero: values differ, 1/1 mismatches [0:1.10000001e-05/0] |}]

let%expect_test "integer and boolean elements are exact" =
  cmp "ints" (i64 [ 3L ] [ 1; 2; 3 ]) (i64 [ 3L ] [ 1; 2; 3 ]);
  cmp "ints differ" (i64 [ 3L ] [ 1; 2; 3 ]) (i64 [ 3L ] [ 1; 2; 4 ]);
  cmp "ints never within a tolerance" ~atol:100. ~rtol:1. (i64 [ 1L ] [ 1000 ])
    (i64 [ 1L ] [ 1001 ]);
  cmp "bools" (bool [ 2L ] [ true; false ]) (bool [ 2L ] [ true; true ]);
  cmp "empty" (f32 [ 0L ] []) (f32 [ 0L ] []);
  [%expect
    {|
    ints: pass, 0/3 mismatches
    ints differ: values differ, 1/3 mismatches [2:4/3]
    ints never within a tolerance: values differ, 1/1 mismatches [0:1001/1000]
    bools: values differ, 1/2 mismatches [1:true/false]
    empty: pass, 0/0 mismatches |}]

let%expect_test "dtype and shape are part of the comparison" =
  cmp "dtype" (f32 [ 2L ] [ 1.; 2. ]) (i64 [ 2L ] [ 1; 2 ]);
  cmp "shape"
    (f32 [ 2L; 2L ] [ 1.; 2.; 3.; 4. ])
    (f32 [ 4L ] [ 1.; 2.; 3.; 4. ]);
  cmp "rank" (f32 [ 1L; 2L ] [ 1.; 2. ]) (f32 [ 2L ] [ 1.; 2. ]);
  [%expect
    {|
    dtype: dtype I64 vs F32, 0/2 mismatches
    shape: shape differs, 0/4 mismatches
    rank: shape differs, 0/2 mismatches |}]

let%expect_test "mismatch coordinates are logical" =
  let e = f32 [ 2L; 3L ] [ 0.; 0.; 0.; 0.; 0.; 0. ] in
  let a = f32 [ 2L; 3L ] [ 0.; 0.; 0.; 0.; 1.; 2. ] in
  cmp "coords" e a;
  (* only the first few mismatches are kept; all are counted *)
  let n = 20 in
  cmp "many"
    (f32 [ Int64.of_int n ] (List.init n (fun _ -> 0.)))
    (f32 [ Int64.of_int n ] (List.init n (fun _ -> 1.)));
  [%expect
    {|
    coords: values differ, 2/6 mismatches [1,1:1/0; 1,2:2/0]
    many: values differ, 20/20 mismatches [0:1/0; 1:1/0; 2:1/0; 3:1/0; 4:1/0; 5:1/0; 6:1/0; 7:1/0] |}]

(* --- reference tensors with arbitrary strides --- *)

let strided ~sizes ~strides ?(offset = 0) values =
  let b = Bytes.create (4 * List.length values) in
  List.iteri
    (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v))
    values;
  {
    Pt2_tensor.dtype = Pt2_dtype.Float32;
    sizes;
    strides;
    storage_offset = offset;
    data = Pt2_storage.of_string (Bytes.to_string b);
  }

let floats (l : F.Logical.t) =
  List.init
    (Int64.to_int (F.Logical.numel l))
    (fun i -> F.Logical.get_float l i)

let gather t =
  match Err.payload (F.Logical.of_pt2 t) with
  | Ok l ->
      Fmt.pr "[%s] %s@."
        (String.concat ";" (List.map Int64.to_string l.shape))
        (String.concat " " (List.map (Printf.sprintf "%g") (floats l)))
  | Error e -> Fmt.pr "%a@." F.Logical.pp_error e

let%expect_test "strided references are gathered into row-major order" =
  let v = [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  print_endline "-- row-major, column-major, offset, negative stride";
  gather (strided ~sizes:[ 2; 3 ] ~strides:[ 3; 1 ] v);
  gather (strided ~sizes:[ 2; 3 ] ~strides:[ 1; 2 ] v);
  gather (strided ~sizes:[ 2; 2 ] ~strides:[ 3; 1 ] ~offset:1 v);
  gather (strided ~sizes:[ 3 ] ~strides:[ -1 ] ~offset:5 v);
  print_endline "-- channels-last storage of a logical NCHW tensor";
  (* N=1, C=2, H=1, W=3 stored NHWC: element (c, w) is at w*2 + c *)
  gather (strided ~sizes:[ 1; 2; 1; 3 ] ~strides:[ 6; 1; 6; 2 ] v);
  print_endline "-- scalar and empty";
  gather (strided ~sizes:[] ~strides:[] [ 7. ]);
  gather (strided ~sizes:[ 0; 3 ] ~strides:[ 3; 1 ] []);
  print_endline "-- reaching outside the storage";
  gather (strided ~sizes:[ 2; 3 ] ~strides:[ 3; 1 ] ~offset:1 v);
  gather (strided ~sizes:[ 3 ] ~strides:[ -1 ] ~offset:1 v);
  gather (strided ~sizes:[ 2 ] ~strides:[ max_int ] v);
  gather (strided ~sizes:[ 2; 3 ] ~strides:[ 3 ] v);
  [%expect
    {|
    -- row-major, column-major, offset, negative stride
    [2;3] 0 1 2 3 4 5
    [2;3] 0 2 4 1 3 5
    [2;2] 1 2 4 5
    [3] 5 4 3
    -- channels-last storage of a logical NCHW tensor
    [1;2;1;3] 0 2 4 1 3 5
    -- scalar and empty
    [] 7
    [0;3]
    -- reaching outside the storage
    strides and offset of dimension 0 reach outside the storage
    strides and offset of dimension 0 reach outside the storage
    strides and offset of dimension 0 reach outside the storage
    strides and offset of dimension 0 reach outside the storage |}]

(* --- content digest, against Python's --- *)

let%expect_test "the content digest line is the producer's" =
  let line name dtype shape =
    match F.Tensor_digest.preamble name dtype shape with
    | Ok s -> Printf.printf "%S\n" s
    | Error _ -> print_endline "rejected"
  in
  line "pixel_values" Dtype.F32 [ 1L; 3L; 256L; 256L ];
  line "scalar" Dtype.I64 [];
  line "mask" Dtype.BOOL [ 0L; 4L ];
  line "we\"ird\\name\n" Dtype.F32 [ 2L ];
  line "naïve" Dtype.F32 [ 2L ];
  [%expect
    {|
    "[\"pixel_values\", \"torch.float32\", [1, 3, 256, 256]]\n"
    "[\"scalar\", \"torch.int64\", []]\n"
    "[\"mask\", \"torch.bool\", [0, 4]]\n"
    "[\"we\\\"ird\\\\name\\n\", \"torch.float32\", [2]]\n"
    rejected |}]
