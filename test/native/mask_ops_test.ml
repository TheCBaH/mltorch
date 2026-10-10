(* The ops attention masks and last-token pooling are built from, at the
   [Graph_ir]/[Eval_direct] level: ge/lt scalar and le tensor comparisons,
   bitwise and, where with a scalar other, new_ones, tanh, the two-index
   gather, and the exact int64 add/slice a position-id buffer goes through.
   Expected values are worked out by hand in the comments. *)

open Graph_ir
open Graph_direct_fixtures

let nan = Float.nan
let inf = Float.infinity

(* Every element, logical row-major, with Bool printed as 0/1. *)
let pp_all ppf tensor =
  let (Tensor.Tensor t) = tensor in
  Format.fprintf ppf "%s %a {"
    (Payload.fmt_name t.Tensor.payload.Payload.fmt)
    Vec6.pp_shape t.Tensor.shape;
  let first = ref true in
  Vec6.iter t.Tensor.shape (fun c ->
      if not !first then Format.fprintf ppf ", ";
      first := false;
      Format.fprintf ppf "%g" (Tensor.read tensor c));
  Format.fprintf ppf "}"

let show label result = Format.printf "%s: %a@." label (pp_result pp_all) result

let floats shape values =
  Tensor.materialize shape (fun c ->
      let flat =
        List.fold_left
          (fun acc a ->
            (acc * (Vec6.get shape a :> int)) + Dim.to_int (Vec6.get c a))
          0 Axis.all
      in
      List.nth values flat)

let longs shape values =
  Tensor.materialize_i64 shape (fun c ->
      let flat =
        List.fold_left
          (fun acc a ->
            (acc * (Vec6.get shape a :> int)) + Dim.to_int (Vec6.get c a))
          0 Axis.all
      in
      List.nth values flat)

(* Build [f] over the named inputs and run it, returning the graph's single
   output. [inputs] are (shape, fmt, tensor) in declaration order. *)
let run ~inputs f =
  let open Err.Syntax in
  let* g =
    lift_build
      Graph_builder.(
        build ~name:"t" ~outputs:(fun r -> [ r ])
        @@
        let rec declare acc i = function
          | [] -> return (List.rev acc)
          | (shape, fmt, _) :: rest ->
              let* id = input ~shape ~fmt ~name:(Printf.sprintf "i%d" i) () in
              declare (id :: acc) (i + 1) rest
        in
        let* ids = declare [] 0 inputs in
        f ids)
  in
  let* env =
    lift_eval
      (Eval_direct.run g
         ~inputs:
           (List.combine g.Graph.inputs (List.map (fun (_, _, t) -> t) inputs)))
  in
  match g.Graph.outputs with
  | id :: _ -> (
      match Tensor_id.Map.find_opt id env with
      | Some t -> Err.return t
      | None -> Err.fail (`Missing_named_tensor "out"))
  | [] -> Err.fail (`Missing_named_tensor "out")

let f32 = Payload.(Fmt F32)
let i64 = Payload.(Fmt I64)
let bool_ = Payload.(Fmt Bool)
let six = s1c 6
let sample = floats six [ 1.; 2.; 3.; nan; -.inf; inf ]

(* x = [1, 2, 3, NaN, -inf, +inf] against 2. NaN is neither above, below nor
   equal, so every comparison is false for it, as in ATen. *)
let%expect_test "scalar comparisons: ge and lt, with NaN and the infinities" =
  let one build_op =
    run
      ~inputs:[ (six, f32, sample) ]
      (function [ x ] -> build_op x | _ -> assert false)
  in
  show "ge 2" (one (Graph_builder.ge_scalar 2.));
  show "lt 2" (one (Graph_builder.lt_scalar 2.));
  [%expect
    {|
    ge 2: bool [C=6] {0, 1, 1, 0, 0, 1}
    lt 2: bool [C=6] {1, 0, 0, 0, 1, 0} |}]

(* a <= b over a column and a row: a_i <= b_j for a = (1,2,3), b = (1,2,3). *)
let%expect_test "le.Tensor broadcasts a column against a row; NaN is false" =
  let col = s 1 1 1 1 3 1 and row = s 1 1 1 1 1 3 in
  let a = floats col [ 1.; 2.; 3. ] and b = floats row [ 1.; 2.; 3. ] in
  show "3x3"
    (run
       ~inputs:[ (col, f32, a); (row, f32, b) ]
       (function [ a; b ] -> Graph_builder.le_tensor a b | _ -> assert false));
  let m = floats six [ 1.; 2.; 3.; nan; -.inf; inf ] in
  let two = floats six [ 2.; 2.; 2.; 2.; 2.; 2. ] in
  show "vs 2"
    (run
       ~inputs:[ (six, f32, m); (six, f32, two) ]
       (function [ a; b ] -> Graph_builder.le_tensor a b | _ -> assert false));
  [%expect
    {|
    3x3: bool [W=3 C=3] {1, 1, 1, 0, 1, 1, 0, 0, 1}
    vs 2: bool [C=6] {1, 1, 0, 0, 1, 0} |}]

let%expect_test "bitwise_and: both must hold; a scalar broadcasts" =
  let four = s1c 4 in
  let a = floats four [ 1.; 0.; 1.; 0. ]
  and b = floats four [ 1.; 1.; 0.; 0. ] in
  show "elementwise"
    (run
       ~inputs:[ (four, bool_, a); (four, bool_, b) ]
       (function
         | [ a; b ] -> Graph_builder.bitwise_and a b | _ -> assert false));
  let one = s1c 1 in
  show "with a true scalar"
    (run
       ~inputs:[ (four, bool_, a); (one, bool_, floats one [ 1. ]) ]
       (function
         | [ a; b ] -> Graph_builder.bitwise_and a b | _ -> assert false));
  show "with a false scalar"
    (run
       ~inputs:[ (four, bool_, a); (one, bool_, floats one [ 0. ]) ]
       (function
         | [ a; b ] -> Graph_builder.bitwise_and a b | _ -> assert false));
  show "an int64 operand is refused"
    (run
       ~inputs:[ (four, i64, longs four [ 1L; 0L; 1L; 0L ]); (four, bool_, b) ]
       (function
         | [ a; b ] -> Graph_builder.bitwise_and a b | _ -> assert false));
  [%expect
    {|
    elementwise: bool [C=4] {1, 0, 0, 0}
    with a true scalar: bool [C=4] {1, 0, 1, 0}
    with a false scalar: bool [C=4] {0, 0, 0, 0}
    an int64 operand is refused: bitwise_and: unsupported mixed dtype, a=i64 b=bool |}]

(* The mask construction: a rank-0 value fans out over the condition. *)
let%expect_test
    "where.ScalarOther: x where the condition holds, else the scalar" =
  let four = s1c 4 and one = s1c 1 in
  let cond = floats four [ 1.; 0.; 1.; 0. ] in
  let neg_max = -3.4028234663852886e+38 in
  show "rank-0 x"
    (run
       ~inputs:[ (four, bool_, cond); (one, f32, floats one [ 5. ]) ]
       (function
         | [ c; x ] -> Graph_builder.where_scalar_other ~condition:c neg_max x
         | _ -> assert false));
  show "full x"
    (run
       ~inputs:
         [ (four, bool_, cond); (four, f32, floats four [ 7.; 8.; 9.; nan ]) ]
       (function
         | [ c; x ] -> Graph_builder.where_scalar_other ~condition:c 0. x
         | _ -> assert false));
  show "an int64 condition is refused"
    (run
       ~inputs:
         [
           (four, i64, longs four [ 1L; 0L; 1L; 0L ]);
           (one, f32, floats one [ 5. ]);
         ]
       (function
         | [ c; x ] -> Graph_builder.where_scalar_other ~condition:c 0. x
         | _ -> assert false));
  [%expect
    {|
    rank-0 x: f32 [C=4] {5, -3.40282e+38, 5, -3.40282e+38}
    full x: f32 [C=4] {7, 0, 9, 0}
    an int64 condition is refused: where_scalar_other: unsupported mixed dtype, a=i64 b=f32 |}]

let%expect_test "new_ones: a bool scalar and a float vector" =
  let ones fmt shape =
    run ~inputs:[] (fun _ ->
        Graph_builder.new_ones { Factory.New_ones.shape; fmt })
  in
  show "bool, rank 0" (ones bool_ (s1c 1));
  show "f32 [3]" (ones f32 (s1c 3));
  [%expect
    {|
    bool, rank 0: bool [C=1] {1}
    f32 [3]: f32 [C=3] {1, 1, 1} |}]

(* Compared with the standard library's tanh, which is correctly rounded to
   within an ulp of double; the engine's float32 result has to land within
   float32 rounding of it. -0 must stay -0. *)
let%expect_test "tanh: values, saturation, NaN and signed zero" =
  let xs = [ 0.; -0.; 0.5; -0.5; 1e-8; 3.; 20.; -20.; nan; inf; -.inf ] in
  let n = List.length xs in
  let shape = s1c n in
  match
    run
      ~inputs:[ (shape, f32, floats shape xs) ]
      (function [ x ] -> Graph_builder.tanh x | _ -> assert false)
  with
  | Error _ -> print_endline "error"
  | Ok y ->
      List.iteri
        (fun i x ->
          let got = Tensor.read y (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:i) in
          let want = Float.tanh (Int32.float_of_bits (Int32.bits_of_float x)) in
          let ok =
            (Float.is_nan got && Float.is_nan want)
            || (got = want && Float.sign_bit got = Float.sign_bit want)
            || Float.abs (got -. want) <= 1e-7 *. Float.max 1. (Float.abs want)
          in
          Format.printf "tanh(%g) = %g %s@." x got
            (if ok then "ok" else "WRONG"))
        xs;
      [%expect
        {|
        tanh(0) = 0 ok
        tanh(-0) = -0 ok
        tanh(0.5) = 0.462117 ok
        tanh(-0.5) = -0.462117 ok
        tanh(1e-08) = 1e-08 ok
        tanh(3) = 0.995055 ok
        tanh(20) = 1 ok
        tanh(-20) = -1 ok
        tanh(nan) = nan ok
        tanh(inf) = 1 ok
        tanh(-inf) = -1 ok |}]

(* --- index.Tensor with two live leading indices --- *)

let pair_params ~self ~i0 ~i1 =
  {
    Index_tensor.Index_pair.self_rank = Rank.of_int self;
    index0_rank = Rank.of_int i0;
    index1_rank = Rank.of_int i1;
  }

let pair ?(self_fmt = f32) ~self_shape ~self_values ~self_rank ~idx0 ~idx1 () =
  let shape0, rank0, v0 = idx0 and shape1, rank1, v1 = idx1 in
  run
    ~inputs:
      [
        (self_shape, self_fmt, floats self_shape self_values);
        (shape0, i64, longs shape0 v0);
        (shape1, i64, longs shape1 v1);
      ]
    (function
      | [ self; index0; index1 ] ->
          Graph_builder.index_pair
            (pair_params ~self:self_rank ~i0:rank0 ~i1:rank1)
            ~self ~index0 ~index1
      | _ -> assert false)

(* self[i, j, k] = 100 i + 10 j + k over [2, 3, 2]; indices (0, 2) and (1, 0)
   give self[0, 2, :] = (20, 21) and self[1, 0, :] = (100, 101): the result is
   the broadcast shape [2] followed by self's trailing axis. *)
let%expect_test "index pair: a rank-3 self keeps its trailing axis" =
  let self_shape = s 1 1 1 2 3 2 in
  let self_values =
    List.init 12 (fun n ->
        let i = n / 6 and j = n / 2 mod 3 and k = n mod 2 in
        float_of_int ((100 * i) + (10 * j) + k))
  in
  let two = s1c 2 in
  show "(0,2) and (1,0)"
    (pair ~self_shape ~self_values ~self_rank:3
       ~idx0:(two, 1, [ 0L; 1L ])
       ~idx1:(two, 1, [ 2L; 0L ])
       ());
  show "negative indices wrap"
    (pair ~self_shape ~self_values ~self_rank:3
       ~idx0:(two, 1, [ -1L; -2L ])
       ~idx1:(two, 1, [ -1L; -3L ])
       ());
  [%expect
    {|
    (0,2) and (1,0): f32 [W=2 C=2] {20, 21, 100, 101}
    negative indices wrap: f32 [W=2 C=2] {120, 121, 0, 1} |}]

(* `mask[batch_idx, kv_idx]` with a [1, 4] bool mask, a [1,1,1,1] batch index
   and a [1,1,1,4] position index: the row, reversed. *)
let%expect_test "index pair: the attention-mask gather keeps a bool self" =
  let self_shape = s1c 4 in
  let one = s 1 1 1 1 1 1 and four = s1c 4 in
  show "reverse"
    (pair ~self_fmt:bool_ ~self_shape ~self_values:[ 1.; 0.; 1.; 1. ]
       ~self_rank:2 ~idx0:(one, 4, [ 0L ])
       ~idx1:(four, 4, [ 3L; 2L; 1L; 0L ])
       ());
  [%expect {| reverse: bool [C=4] {1, 1, 0, 1} |}]

let%expect_test "index pair: broadcast mismatch and a bad rank are errors" =
  let self_shape = s 1 1 1 2 3 2 in
  let self_values = List.init 12 float_of_int in
  show "indices of extent 2 and 3"
    (pair ~self_shape ~self_values ~self_rank:3
       ~idx0:(s1c 2, 1, [ 0L; 1L ])
       ~idx1:(s1c 3, 1, [ 0L; 1L; 2L ])
       ());
  show "self of rank 1"
    (pair ~self_shape:(s1c 4) ~self_values:[ 1.; 2.; 3.; 4. ] ~self_rank:1
       ~idx0:(s1c 1, 1, [ 0L ])
       ~idx1:(s1c 1, 1, [ 0L ])
       ());
  show "an index rank that its shape contradicts"
    (pair ~self_shape ~self_values ~self_rank:3
       ~idx0:(s 1 1 1 1 2 2, 1, [ 0L; 1L; 0L; 1L ])
       ~idx1:(s1c 2, 1, [ 0L; 1L ])
       ());
  [%expect
    {|
    indices of extent 2 and 3: incompatible broadcast extents on axis W: 2 vs 3
    self of rank 1: index.Tensor: two leading indices need a self of rank >= 2 whose remaining axes plus the rank-1 broadcast index fit 6 axes (self has rank 1)
    an index rank that its shape contradicts: index.Tensor: index declared rank 1, but its own axis W has extent 2 (must be 1, outside a rank-1 tensor's own real axes) |}]

let%expect_test
    "index pair: an out-of-range value raises, never reads out of bounds" =
  (try
     show "index 2 into extent 2"
       (pair ~self_shape:(s 1 1 1 2 3 2)
          ~self_values:(List.init 12 float_of_int)
          ~self_rank:3
          ~idx0:(s1c 2, 1, [ 0L; 2L ])
          ~idx1:(s1c 2, 1, [ 0L; 1L ])
          ())
   with Err.Exn.E e -> Format.printf "raised: %a@." Err.Exn.pp_kind e);
  [%expect {| raised: gather index 2 out of range [-2, 1] |}]

(* --- int64 stays exact --- *)

let pp_longs ppf tensor =
  let (Tensor.Tensor t) = tensor in
  Format.fprintf ppf "%s %a {"
    (Payload.fmt_name t.Tensor.payload.Payload.fmt)
    Vec6.pp_shape t.Tensor.shape;
  let first = ref true in
  Vec6.iter t.Tensor.shape (fun c ->
      if not !first then Format.fprintf ppf ", ";
      first := false;
      Format.fprintf ppf "%Ld"
        (Err.or_raise
           ~pp_error:(fun fmt _ -> Fmt.string fmt "not int64")
           (Tensor.read_i64_at6 tensor (fun a -> (Vec6.get c a :> int)))));
  Format.fprintf ppf "}"

(* 2^53 + 1 is not a float64: a trip through the float domain would give
   2^53 + 2 after the add. *)
let%expect_test "int64 add_scalar and slice never touch the float domain" =
  let three = s1c 3 in
  let t = longs three [ 9007199254740993L; 5L; -7L ] in
  let show_long label r =
    Format.printf "%s: %a@." label (pp_result pp_longs) r
  in
  show_long "+ 1"
    (run
       ~inputs:[ (three, i64, t) ]
       (function [ x ] -> Graph_builder.add_scalar 1. x | _ -> assert false));
  show_long "slice [1, 3)"
    (run
       ~inputs:[ (three, i64, t) ]
       (function
         | [ x ] ->
             Graph_builder.slice
               {
                 Split.Slice.axis = Axis.C;
                 start = Dim.fence 1;
                 stop = Dim.fence 3;
                 step = Op_config.Pos.of_int 1;
               }
               x
         | _ -> assert false));
  show "+ 0.5 leaves the integer path"
    (run
       ~inputs:[ (three, i64, t) ]
       (function [ x ] -> Graph_builder.add_scalar 0.5 x | _ -> assert false));
  [%expect
    {|
    + 1: i64 [C=3] {9007199254740994, 6, -6}
    slice [1, 3): i64 [C=2] {5, -7}
    + 0.5 leaves the integer path: f32 [C=3] {9.0072e+15, 5.5, -6.5} |}]

(* --- argmax and the int32 cast: TinyCLIP's last-token pooling --- *)

let argmax_params axis keepdim = { Reduce.Argmax.axis; keepdim }

let%expect_test "argmax: first maximum, NaN wins, keepdim, int64 result" =
  let shape = s 1 1 1 1 3 4 in
  let x =
    floats shape
      [ 1.; 5.; 5.; 2.; nan; 1.; nan; 0.; -.inf; -.inf; -.inf; -.inf ]
  in
  let go axis keepdim =
    run
      ~inputs:[ (shape, f32, x) ]
      (function
        | [ x ] -> Graph_builder.argmax (argmax_params axis keepdim) x
        | _ -> assert false)
  in
  let show_long label r =
    Format.printf "%s: %a@." label (pp_result pp_longs) r
  in
  (* Rows (W): [1 5 5 2] -> 1 (the first 5); [nan 1 nan 0] -> 2: a NaN is the
     maximum, but this engine reports the LAST of several (the pooling
     predicate), where ATen reports the first -- a known difference, pinned
     here so it cannot change unnoticed; [-inf x4] -> 0 (the first of a tie). *)
  show_long "along C" (go Axis.C false);
  show_long "along C, keepdim" (go Axis.C true);
  (* Columns (W): c0 = [1 nan -inf] -> 1; c1 = [5 1 -inf] -> 0;
     c2 = [5 nan -inf] -> 1; c3 = [2 0 -inf] -> 0. *)
  show_long "along W" (go Axis.W false);
  [%expect
    {|
    along C: i64 [C=3] {1, 2, 0}
    along C, keepdim: i64 [W=3 C=1] {1, 2, 0}
    along W: i64 [C=4] {1, 0, 1, 0} |}]

let%expect_test "to_copy int32 keeps in-range values and refuses the rest" =
  let three = s1c 3 in
  let cast fmt t =
    run
      ~inputs:[ (three, fmt, t) ]
      (function
        | [ x ] -> Graph_builder.to_copy Pointwise.To_copy.Int x
        | _ -> assert false)
  in
  let show_long label r =
    Format.printf "%s: %a@." label (pp_result pp_longs) r
  in
  show_long "i64 in range"
    (cast i64 (longs three [ 49407L; -2147483648L; 2147483647L ]));
  show_long "bool"
    (cast bool_
       (Tensor.materialize_bool three (fun c ->
            Dim.to_int (Vec6.get c Axis.C) <> 1)));
  show_long "f32 truncates" (cast f32 (floats three [ 2.9; -2.9; 7. ]));
  let try_cast label fmt t =
    try show_long label (cast fmt t)
    with Err.Exn.E e ->
      Format.printf "%s: raised: %a@." label Err.Exn.pp_kind e
  in
  try_cast "i64 one past the top" i64 (longs three [ 0L; 0L; 2147483648L ]);
  try_cast "i64 one below the bottom" i64 (longs three [ -2147483649L; 0L; 0L ]);
  [%expect
    {|
    i64 in range: i64 [C=3] {49407, -2147483648, 2147483647}
    bool: i64 [C=3] {1, 0, 1}
    f32 truncates: i64 [C=3] {2, -2, 7}
    i64 one past the top: raised: to_copy int32: 2147483648 is outside [-2^31, 2^31)
    i64 one below the bottom: raised: to_copy int32: -2147483649 is outside [-2^31, 2^31) |}]

let%expect_test "exp: values, the infinities, NaN and float32 overflow" =
  let xs = [ 0.; 1.; -1.; -.inf; inf; nan; 100.; -100. ] in
  let shape = s1c (List.length xs) in
  show "exp"
    (run
       ~inputs:[ (shape, f32, floats shape xs) ]
       (function [ x ] -> Graph_builder.exp x | _ -> assert false));
  (* e = 2.71828 and 1/e = 0.367879; exp(100) = 2.7e43 exceeds float32 and
     stores as +inf; exp(-100) = 3.7e-44 is a float32 denormal. *)
  [%expect
    {| exp: f32 [C=8] {1, 2.71828, 0.367879, 0, inf, nan, inf, 3.78351e-44} |}]

(* --- the ops T5's relative-position buckets are built from --- *)

let%expect_test "log: values, zero, negative, infinity and NaN" =
  let xs = [ 1.; 8.; 0.; -1.; inf; nan ] in
  let shape = s1c (List.length xs) in
  show "log"
    (run
       ~inputs:[ (shape, f32, floats shape xs) ]
       (function [ x ] -> Graph_builder.log x | _ -> assert false));
  (* ln 8 = 2.07944; ln 0 = -inf; ln of a negative is NaN. *)
  [%expect {| log: f32 [C=6] {0, 2.07944, -inf, nan, inf, nan} |}]

let%expect_test "min.other: the lesser, NaN wins, and int64 is exact" =
  let four = s1c 4 in
  let a = floats four [ 1.; 5.; nan; 2. ]
  and b = floats four [ 3.; 2.; 1.; nan ] in
  show "float"
    (run
       ~inputs:[ (four, f32, a); (four, f32, b) ]
       (function [ a; b ] -> Graph_builder.min_other a b | _ -> assert false));
  let big = 9007199254740993L in
  let show_long label r =
    Format.printf "%s: %a@." label (pp_result pp_longs) r
  in
  show_long "int64 past 2^53"
    (run
       ~inputs:
         [
           (four, i64, longs four [ big; 3L; -5L; 0L ]);
           (four, i64, longs four [ 9007199254740992L; 4L; -7L; 0L ]);
         ]
       (function [ a; b ] -> Graph_builder.min_other a b | _ -> assert false));
  show "a mixed int64/float pair is refused"
    (run
       ~inputs:[ (four, i64, longs four [ 1L; 2L; 3L; 4L ]); (four, f32, b) ]
       (function [ a; b ] -> Graph_builder.min_other a b | _ -> assert false));
  [%expect
    {|
    float: f32 [C=4] {1, 2, nan, nan}
    int64 past 2^53: i64 [C=4] {9007199254740992, 3, -7, 0}
    a mixed int64/float pair is refused: min_other: unsupported mixed dtype, a=i64 b=f32 |}]

let%expect_test "where.self selects between tensors; int64 branches stay exact"
    =
  let four = s1c 4 in
  let cond = floats four [ 1.; 0.; 1.; 0. ] in
  show "float"
    (run
       ~inputs:
         [
           (four, bool_, cond);
           (four, f32, floats four [ 1.; 2.; 3.; 4. ]);
           (four, f32, floats four [ 10.; 20.; 30.; 40. ]);
         ]
       (function
         | [ c; x; y ] -> Graph_builder.where_self ~condition:c x y
         | _ -> assert false));
  let big = 9007199254740993L in
  let show_long label r =
    Format.printf "%s: %a@." label (pp_result pp_longs) r
  in
  show_long "int64 past 2^53, a rank-0 branch broadcast"
    (run
       ~inputs:
         [
           (four, bool_, cond);
           (s1c 1, i64, longs (s1c 1) [ big ]);
           (four, i64, longs four [ 1L; 2L; 3L; 4L ]);
         ]
       (function
         | [ c; x; y ] -> Graph_builder.where_self ~condition:c x y
         | _ -> assert false));
  [%expect
    {|
    float: f32 [C=4] {1, 20, 3, 40}
    int64 past 2^53, a rank-0 branch broadcast: i64 [C=4] {9007199254740993, 2, 9007199254740993, 4} |}]

let%expect_test "full_like keeps the format; a fractional int64 fill is refused"
    =
  let three = s1c 3 in
  let show_long label r =
    Format.printf "%s: %a@." label (pp_result pp_longs) r
  in
  show "float"
    (run
       ~inputs:[ (three, f32, floats three [ 1.; 2.; 3. ]) ]
       (function [ x ] -> Graph_builder.full_like 2.5 x | _ -> assert false));
  show_long "int64 15"
    (run
       ~inputs:[ (three, i64, longs three [ 1L; 2L; 3L ]) ]
       (function [ x ] -> Graph_builder.full_like 15. x | _ -> assert false));
  show "bool zeros"
    (run
       ~inputs:[ (three, bool_, Tensor.materialize_bool three (fun _ -> true)) ]
       (function [ x ] -> Graph_builder.full_like 0. x | _ -> assert false));
  (try
     show_long "int64 2.5"
       (run
          ~inputs:[ (three, i64, longs three [ 1L; 2L; 3L ]) ]
          (function
            | [ x ] -> Graph_builder.full_like 2.5 x | _ -> assert false))
   with Err.Exn.E e ->
     Format.printf "int64 2.5: raised: %a@." Err.Exn.pp_kind e);
  [%expect
    {|
    float: f32 [C=3] {2.5, 2.5, 2.5}
    int64 15: i64 [C=3] {15, 15, 15}
    bool zeros: bool [C=3] {0, 0, 0}
    int64 2.5: raised: full_like: 2.5 is not a whole number for an int64 tensor |}]

let%expect_test "an int64 times a whole scalar stays int64 and exact" =
  let three = s1c 3 in
  let t = longs three [ 4611686018427387905L; 3L; -2L ] in
  let show_long label r =
    Format.printf "%s: %a@." label (pp_result pp_longs) r
  in
  (* 2^62 + 1 times 2 is 2^63 + 2, which wraps to -2^63 + 2 in int64. *)
  show_long "* 2"
    (run
       ~inputs:[ (three, i64, t) ]
       (function [ x ] -> Graph_builder.mul_scalar 2. x | _ -> assert false));
  show "* 0.5 promotes to float"
    (run
       ~inputs:[ (three, i64, longs three [ 3L; 4L; -5L ]) ]
       (function [ x ] -> Graph_builder.mul_scalar 0.5 x | _ -> assert false));
  [%expect
    {|
    * 2: i64 [C=3] {-9223372036854775806, 6, -4}
    * 0.5 promotes to float: f32 [C=3] {1.5, 2, -2.5} |}]

(* The cast T5's bucket table relies on: log 0 = -inf cast to int64 and then
   discarded by a where. Checked rejects it; the aarch64 conversion saturates. *)
let%expect_test "float to int64: checked rejects, saturating follows aarch64" =
  let xs = [ 2.9; -2.9; inf; -.inf; nan; 1e30 ] in
  let shape = s1c (List.length xs) in
  let cast () =
    run
      ~inputs:[ (shape, f32, floats shape xs) ]
      (function
        | [ x ] -> Graph_builder.to_copy Pointwise.To_copy.Long x
        | _ -> assert false)
  in
  (try Format.printf "checked: %a@." (pp_result pp_longs) (cast ())
   with Err.Exn.E e -> Format.printf "checked: raised: %a@." Err.Exn.pp_kind e);
  Format.printf "saturating: %a@." (pp_result pp_longs)
    (Direct.with_float_to_int Direct.Saturating cast);
  [%expect
    {|
    checked: raised: Float-to-I64 cast of an infinite value
    saturating: i64 [C=6] {2, -2, 9223372036854775807, -9223372036854775808, 0, 9223372036854775807} |}]

let%expect_test "log1p: accurate for a tiny x, and the edges" =
  let xs = [ 0.; 1e-10; -1e-10; 1.; -1.; -2.; inf; nan; 1e-3 ] in
  let shape = s1c (List.length xs) in
  (match
     run
       ~inputs:[ (shape, f32, floats shape xs) ]
       (function [ x ] -> Graph_builder.log1p x | _ -> assert false)
   with
  | Error _ -> print_endline "error"
  | Ok y ->
      List.iteri
        (fun i x ->
          let got = Tensor.read y (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:i) in
          let x32 = Int32.float_of_bits (Int32.bits_of_float x) in
          (* [Float.log1p] is the correctly rounded reference. *)
          let want = Float.log1p x32 in
          let ok =
            (Float.is_nan got && Float.is_nan want)
            || got = want
            || Float.abs (got -. want) <= 1.2e-7 *. Float.abs want
          in
          Format.printf "log1p(%g) = %g %s@." x got
            (if ok then "ok" else "WRONG"))
        xs);
  [%expect
    {|
    log1p(0) = 0 ok
    log1p(1e-10) = 1e-10 ok
    log1p(-1e-10) = -1e-10 ok
    log1p(1) = 0.693147 ok
    log1p(-1) = -inf ok
    log1p(-2) = nan ok
    log1p(inf) = inf ok
    log1p(nan) = nan ok
    log1p(0.001) = 0.0009995 ok |}]

(* --- exact int64 data movement: clone, expand, repeat, repeat_interleave --- *)

let%expect_test "int64 data movement keeps the dtype and values past 2^53" =
  let big = 9007199254740993L in
  let show_long label r =
    Format.printf "%s: %a@." label (pp_result pp_longs) r
  in
  let two = s1c 2 in
  let t = longs two [ big; -7L ] in
  show_long "clone"
    (run
       ~inputs:[ (two, i64, t) ]
       (function [ x ] -> Graph_builder.clone x | _ -> assert false));
  show_long "expand [2] -> [3, 2]"
    (run
       ~inputs:[ (two, i64, t) ]
       (function
         | [ x ] ->
             Graph_builder.expand { Pointwise.Expand.size = s 1 1 1 1 3 2 } x
         | _ -> assert false));
  show_long "repeat x2"
    (run
       ~inputs:[ (two, i64, t) ]
       (function
         | [ x ] ->
             Graph_builder.repeat { Repeat.Repeat.repeats = s 1 1 1 1 1 2 } x
         | _ -> assert false));
  show_long "repeat_interleave x2"
    (run
       ~inputs:[ (two, i64, t) ]
       (function
         | [ x ] ->
             Graph_builder.repeat_interleave
               {
                 Repeat.RepeatInterleave.axis = Axis.C;
                 repeats = Op_config.Pos.of_int 2;
               }
               x
         | _ -> assert false));
  [%expect
    {|
    clone: i64 [C=2] {9007199254740993, -7}
    expand [2] -> [3, 2]: i64 [W=3 C=2] {9007199254740993, -7, 9007199254740993, -7, 9007199254740993, -7}
    repeat x2: i64 [C=4] {9007199254740993, -7, 9007199254740993, -7}
    repeat_interleave x2: i64 [C=4] {9007199254740993, 9007199254740993, -7, -7} |}]
