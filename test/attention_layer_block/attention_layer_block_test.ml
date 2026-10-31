(* The layer's reference against its parts run by hand on the same operands,
   each part built here in a layout of its own (one row a norm, one sequence a
   projection, one head an attention over copied dense operands), so a stride
   the layer derives wrong shows; and its refusals and name. *)

module L = Attention_layer_block
module A = Attention_block
module C = Contraction
module N = Norm_block
module P = Attention_block.Precision

let i64 = Int64.of_int
let get_l d = Err.or_raise ~pp_error:L.pp_error d
let ok d = Result.get_ok (Err.payload d)
let nan32 = Int32.float_of_bits 0x7FC00000l
let r32 x = if Float.is_nan x then nan32 else F32_math.round32 x
let eps = F32_math.round32 1e-5
let lcg = ref 2718

let uniform scale =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 2000. *. scale)

let mask_value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  match !lcg / 13 mod 9 with
  | 0 -> Float.neg_infinity
  | n -> float_of_int (n - 4) /. 4.

type operands = {
  x : float array;
  nw : float array;
  nb : float array;
  w : float array array;  (** q, k, v, o *)
  bias : float array array;
  gamma : float array;
  cos : float array;  (** [tokens][d], shared by the sequences *)
  sin : float array;
  mask : float array;  (** [batch][tokens][tokens], shared by the heads *)
}

let operands ~bf16 ~batch ~tokens ~width ~d =
  let fill n s = Array.init n (fun _ -> uniform s) in
  let weight () =
    let w = fill (width * width) (1. /. sqrt (float_of_int width)) in
    if bf16 then Array.map A.bf16_round w else w
  in
  {
    x = fill (batch * tokens * width) 2.;
    nw = fill width 1.5;
    nb = fill width 0.5;
    w = Array.init 4 (fun _ -> weight ());
    bias = Array.init 4 (fun _ -> fill width 0.25);
    gamma = fill width 1.;
    cos = fill (tokens * d) 1.;
    sin = fill (tokens * d) 1.;
    mask = Array.init (batch * tokens * tokens) (fun _ -> mask_value ());
  }

(* The composition by hand. *)
let by_hand ~precision ~batch ~tokens ~width ~heads ~norm ~(flags : L.Flags.t)
    ~rope ~masked ~block_keys (o : operands) =
  let d = width / heads and e = width in
  let bf16 = precision = P.Bf16_paired in
  let operand v = if bf16 then A.bf16_round v else v in
  let norm_row precision (n : L.Norm_spec.t) row =
    let desc =
      ok
        (N.make ~kind:n.L.Norm_spec.kind ~rows:1L ~width:(i64 e)
           ~eps:n.L.Norm_spec.eps ~weight:n.L.Norm_spec.weight
           ~bias:n.L.Norm_spec.bias
           ~x:{ N.Strides.row = 0L; col = 1L }
           ~y:{ N.Strides.row = 0L; col = 1L }
           ~precision)
    in
    let out = Array.make e 0. in
    N.eval desc
      ~x:(fun c -> row.(Int64.to_int c))
      ~weight:(fun c -> o.nw.(Int64.to_int c))
      ~bias:(fun c -> o.nb.(Int64.to_int c))
      ~write:(fun c v -> out.(Int64.to_int c) <- v);
    out
  in
  let rows a = Array.init (batch * tokens) (fun r -> Array.sub a (r * e) e) in
  let flat rs = Array.concat (Array.to_list rs) in
  let x_rows = rows o.x in
  let u =
    match norm with
    | Some (n : L.Norm_spec.t) when n.L.Norm_spec.placement = L.Placement.Pre ->
        Array.map (norm_row precision n) x_rows
    | _ -> Array.map (Array.map operand) x_rows
  in
  (* per sequence: [tokens][e] times W^T, then the bias *)
  let project which (input : float array array) biased =
    let desc =
      ok
        (C.make ~batch:(i64 batch) ~m:(i64 tokens) ~n:(i64 e) ~k:(i64 e)
           ~a:{ C.Operand.batch = i64 (tokens * e); outer = i64 e; k = 1L }
           ~b:{ C.Operand.batch = 0L; outer = i64 e; k = 1L }
           ~c:{ C.Output.batch = i64 (tokens * e); row = i64 e; col = 1L }
           ~format:(P.format precision) ~seed:C.Seed.Zero
           ~numerics:(P.numerics precision))
    in
    let a = flat input and out = Array.make (batch * tokens * e) 0. in
    C.eval desc
      ~a:(fun i -> a.(Int64.to_int i))
      ~b:(fun i -> o.w.(which).(Int64.to_int i))
      ~c:(fun _ -> 0.)
      ~write:(fun i v -> out.(Int64.to_int i) <- v);
    rows
      (if biased then
         Array.mapi (fun i v -> r32 (v +. o.bias.(which).(i mod e))) out
       else out)
  in
  let q = project 0 u flags.L.Flags.q_bias
  and k = project 1 u flags.L.Flags.k_bias
  and v = project 2 u flags.L.Flags.v_bias in
  (* one head of one row: cat(-z2, z1) *)
  let rotate (rs : float array array) =
    if rope then
      Array.mapi
        (fun r row ->
          let tok = r mod tokens in
          Array.concat
            (List.init heads (fun h ->
                 let z = Array.sub row (h * d) d in
                 let half = d / 2 in
                 let rot =
                   Array.append
                     (Array.map
                        (fun x -> r32 (x *. -1.))
                        (Array.sub z half half))
                     (Array.sub z 0 half)
                 in
                 Array.init d (fun f ->
                     r32
                       (r32 (z.(f) *. o.cos.((tok * d) + f))
                       +. r32 (rot.(f) *. o.sin.((tok * d) + f)))))))
        rs
    else rs
  in
  let q = rotate q and k = rotate k in
  let q = Array.map (Array.map operand) q
  and k = Array.map (Array.map operand) k
  and v = Array.map (Array.map operand) v in
  let att = Array.init (batch * tokens) (fun _ -> Array.make e 0.) in
  for b = 0 to batch - 1 do
    for h = 0 to heads - 1 do
      let dense rs =
        Array.init (tokens * d) (fun i ->
            rs.((b * tokens) + (i / d)).((h * d) + (i mod d)))
      in
      let qd = dense q and kd = dense k and vd = dense v in
      let s = { A.Strides.batch = 0L; head = 0L; row = i64 d; col = 1L } in
      let desc =
        ok
          (A.make ~batch:1L ~heads:1L ~tq:(i64 tokens) ~tk:(i64 tokens)
             ~d:(i64 d) ~q:s ~k:s ~v:s ~o:s
             ?mask:
               (if masked then
                  Some
                    {
                      A.Strides.batch = 0L;
                      head = 0L;
                      row = i64 tokens;
                      col = 1L;
                    }
                else None)
             ~scale:(A.Scale.default ~d:(i64 d))
             ~block_keys:(i64 block_keys) ~precision ())
      in
      A.eval desc
        ~q:(fun i -> qd.(Int64.to_int i))
        ~k:(fun i -> kd.(Int64.to_int i))
        ~v:(fun i -> vd.(Int64.to_int i))
        ~mask:(fun i -> o.mask.((b * tokens * tokens) + Int64.to_int i))
        ~write:(fun i x ->
          let i = Int64.to_int i in
          att.((b * tokens) + (i / d)).((h * d) + (i mod d)) <- x)
    done
  done;
  let att = Array.map (Array.map operand) att in
  let out = project 3 att flags.L.Flags.o_bias in
  let res =
    Array.mapi
      (fun r row ->
        Array.mapi
          (fun c v ->
            let v =
              if flags.L.Flags.layer_scale then r32 (v *. o.gamma.(c)) else v
            in
            r32 (v +. x_rows.(r).(c)))
          row)
      out
  in
  flat
    (match norm with
    | Some n when n.L.Norm_spec.placement = L.Placement.Post ->
        (* a Post norm's output is the layer's: binary32 even for bfloat16 *)
        Array.map (norm_row (if bf16 then P.F32_ordered else precision) n) res
    | _ -> res)

let layer_eval (t : L.t) (o : operands) ~d ~tokens =
  let cells = Array.length o.x in
  let out = Array.make cells 0. in
  let at a i = a.(Int64.to_int i) in
  L.eval t
    {
      L.Readers.x = at o.x;
      norm_weight = at o.nw;
      norm_bias = at o.nb;
      wq = at o.w.(0);
      bq = at o.bias.(0);
      wk = at o.w.(1);
      bk = at o.bias.(1);
      wv = at o.w.(2);
      bv = at o.bias.(2);
      wo = at o.w.(3);
      bo = at o.bias.(3);
      gamma = at o.gamma;
      cos = at o.cos;
      sin = at o.sin;
      mask = at o.mask;
    }
    ~write:(fun i v -> out.(Int64.to_int i) <- v);
  ignore (d, tokens);
  out

let pre_layer =
  {
    L.Norm_spec.kind = N.Kind.Layer;
    placement = L.Placement.Pre;
    eps;
    weight = true;
    bias = true;
  }

let configs =
  [
    ("bare", None, L.Flags.none, false, false);
    ("pre layer, all, masked", Some pre_layer, L.Flags.all, false, true);
    ( "post layer, all",
      Some { pre_layer with L.Norm_spec.placement = L.Placement.Post },
      L.Flags.all,
      false,
      false );
    ( "pre rms, rope, masked",
      Some { pre_layer with L.Norm_spec.kind = N.Kind.Rms; bias = false },
      { L.Flags.none with q_bias = true; k_bias = true },
      true,
      true );
    ("no norm, all, rope", None, L.Flags.all, true, false);
  ]

let%expect_test "the reference is its parts composed" =
  let batch = 2 and tokens = 7 and width = 8 and heads = 2 and block_keys = 4 in
  let d = width / heads in
  List.iter
    (fun precision ->
      List.iter
        (fun (name, norm, flags, rope, masked) ->
          let t =
            get_l
              (L.make ~batch:(i64 batch) ~tokens:(i64 tokens) ~width:(i64 width)
                 ~heads:(i64 heads) ?norm ~flags
                 ?rope:
                   (if rope then
                      Some { L.Rope.batch = 0L; row = i64 d; col = 1L }
                    else None)
                 ?mask:
                   (if masked then
                      Some
                        {
                          A.Strides.batch = i64 (tokens * tokens);
                          head = 0L;
                          row = i64 tokens;
                          col = 1L;
                        }
                    else None)
                 ~scale:(A.Scale.default ~d:(i64 d))
                 ~block_keys:(i64 block_keys) ~precision ())
          in
          let o =
            operands ~bf16:(precision = P.Bf16_paired) ~batch ~tokens ~width ~d
          in
          let got = layer_eval t o ~d ~tokens
          and want =
            by_hand ~precision ~batch ~tokens ~width ~heads ~norm ~flags ~rope
              ~masked ~block_keys o
          in
          Fmt.pr "%s, %s: %s@." (P.name precision) name
            (if Array.for_all2 Core.Float_bits.equal_portable got want then
               "equals its parts"
             else "DIFFERS from its parts"))
        configs)
    P.all;
  [%expect
    {|
    bf16_paired, bare: equals its parts
    bf16_paired, pre layer, all, masked: equals its parts
    bf16_paired, post layer, all: equals its parts
    bf16_paired, pre rms, rope, masked: equals its parts
    bf16_paired, no norm, all, rope: equals its parts
    f32_fused, bare: equals its parts
    f32_fused, pre layer, all, masked: equals its parts
    f32_fused, post layer, all: equals its parts
    f32_fused, pre rms, rope, masked: equals its parts
    f32_fused, no norm, all, rope: equals its parts
    f32_ordered, bare: equals its parts
    f32_ordered, pre layer, all, masked: equals its parts
    f32_ordered, post layer, all: equals its parts
    f32_ordered, pre rms, rope, masked: equals its parts
    f32_ordered, no norm, all, rope: equals its parts |}]

let%expect_test "refusals" =
  let pr ?norm ?rope ?(width = 8L) ?(heads = 2L) ?(block_keys = 4L) () =
    match
      Err.payload
        (L.make ~batch:1L ~tokens:4L ~width ~heads ?norm ~flags:L.Flags.none
           ?rope ~scale:0.5 ~block_keys ~precision:P.F32_ordered ())
    with
    | Ok t -> Fmt.pr "%a@." L.pp t
    | Error e -> Fmt.pr "%a@." L.pp_error e
  in
  pr ~width:9L ();
  pr ~heads:0L ();
  pr ~width:6L ~heads:2L ~rope:{ L.Rope.batch = 0L; row = 3L; col = 1L } ();
  pr ~rope:{ L.Rope.batch = 0L; row = -4L; col = 1L } ();
  pr ~block_keys:6L ();
  pr ~norm:{ pre_layer with L.Norm_spec.kind = N.Kind.Rms } ();
  pr ~norm:{ pre_layer with L.Norm_spec.eps = 0.1 } ();
  pr ();
  [%expect
    {|
    attention layer width 9 is not a multiple of 2 heads
    attention layer extent heads = 0
    a rotary embedding over an odd head dimension 3
    rotary stride row = -4
    attention block of 6 keys, not a positive multiple of four
    an RmsNorm has no bias
    norm eps 0x1.999999999999ap-4, not a finite non-negative binary32
    attention_layer batch=1 tokens=4 width=8 heads=2 norm=none flags=[] rope=none attend batch=1 heads=2 tq=4 tk=4 d=4 q=(32,4,8,1) k=(32,4,8,1) v=(32,4,8,1) o=(32,4,8,1) mask=none scale=0x1p-1 block_keys=4 precision=f32_ordered |}]

let%expect_test "a name rebuilds its descriptor" =
  List.iter
    (fun (name, norm, flags, rope, masked) ->
      let t =
        get_l
          (L.make ~batch:2L ~tokens:7L ~width:8L ~heads:2L ?norm ~flags
             ?rope:
               (if rope then Some { L.Rope.batch = 28L; row = 4L; col = 1L }
                else None)
             ?mask:
               (if masked then
                  Some { A.Strides.batch = 49L; head = 0L; row = 7L; col = 1L }
                else None)
             ~scale:0.5 ~block_keys:4L ~precision:P.F32_fused ())
      in
      Fmt.pr "%s: %s %b@." name (L.name t) (L.of_name (L.name t) = Some t))
    configs;
  Fmt.pr "%s@."
    (L.symbol
       (get_l
          (L.make ~batch:1L ~tokens:4L ~width:8L ~heads:2L ~flags:L.Flags.none
             ~scale:0.5 ~block_keys:4L ~precision:P.F32_ordered ())));
  Fmt.pr "%b@."
    (L.of_name
       "attention_layer.b1.t4.e8.h2.nonorm.fqq.norope.nomask.x3f000000.n4.pf32_ordered"
    = None);
  [%expect
    {|
    bare: attention_layer.b2.t7.e8.h2.nonorm.f.norope.nomask.x3f000000.n4.pf32_fused true
    pre layer, all, masked: attention_layer.b2.t7.e8.h2.nlayer_pre_3727c5ac_wb.fqkvog.norope.m49_0_7_1.x3f000000.n4.pf32_fused true
    post layer, all: attention_layer.b2.t7.e8.h2.nlayer_post_3727c5ac_wb.fqkvog.norope.nomask.x3f000000.n4.pf32_fused true
    pre rms, rope, masked: attention_layer.b2.t7.e8.h2.nrms_pre_3727c5ac_w.fqk.r28_4_1.m49_0_7_1.x3f000000.n4.pf32_fused true
    no norm, all, rope: attention_layer.b2.t7.e8.h2.nonorm.fqkvog.r28_4_1.nomask.x3f000000.n4.pf32_fused true
    mlt_attention_layer_c5ec68e6117d7269
    true |}]
