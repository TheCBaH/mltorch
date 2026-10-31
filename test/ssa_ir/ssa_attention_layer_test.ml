(* A granted attention layer's SSA sequence, its attention expanded and the
   program narrowed to binary32, gives the bits [Attention_layer_block.eval]
   gives. The cases and their inputs are shared with the C, Wasm and
   JavaScript suites, which run the same program through their backend. *)

open Ssa_ir
module F = Ssa_fixtures
module L = Attention_layer_block
module A = Attention_block
module P = Attention_block.Precision

let i64 = Int64.of_int

module Rope_form = struct
  type t = Batched | No_rope | Shared
end

module Case = struct
  type t = {
    name : string;
    batch : int;
    tokens : int;
    width : int;
    heads : int;
    block_keys : int;
    norm : L.Norm_spec.t option;
    flags : L.Flags.t;
    rope : Rope_form.t;
    mask : bool;
  }
end

let eps = F32_math.round32 1e-5

let layer placement ~weight ~bias =
  { L.Norm_spec.kind = Norm_block.Kind.Layer; placement; eps; weight; bias }

let rms placement =
  {
    L.Norm_spec.kind = Norm_block.Kind.Rms;
    placement;
    eps;
    weight = true;
    bias = false;
  }

let qkv_bias = { L.Flags.none with q_bias = true; k_bias = true; v_bias = true }

let case ?(batch = 2) ?(tokens = 5) ?(width = 8) ?(heads = 2) ?(block_keys = 4)
    ?norm ?(flags = L.Flags.none) ?(rope = Rope_form.No_rope) ?(mask = false)
    name =
  {
    Case.name;
    batch;
    tokens;
    width;
    heads;
    block_keys;
    norm;
    flags;
    rope;
    mask;
  }

let cases =
  [
    case "bare";
    case "pre layer, all flags, masked"
      ~norm:(layer L.Placement.Pre ~weight:true ~bias:true)
      ~flags:L.Flags.all ~mask:true;
    case "post layer, all flags"
      ~norm:(layer L.Placement.Post ~weight:true ~bias:true)
      ~flags:L.Flags.all;
    case "pre rms, qkv biases, shared rope, masked" ~norm:(rms L.Placement.Pre)
      ~flags:qkv_bias ~rope:Rope_form.Shared ~mask:true;
    case "pre layer w, all flags, batched rope"
      ~norm:(layer L.Placement.Pre ~weight:true ~bias:false)
      ~flags:L.Flags.all ~rope:Rope_form.Batched;
    case "no norm, all flags, shared rope, masked, 7 tokens" ~tokens:7
      ~flags:L.Flags.all ~rope:Rope_form.Shared ~mask:true;
    case "post rms, one head of 12, 9 tokens" ~heads:1 ~width:12 ~tokens:9
      ~block_keys:8 ~norm:(rms L.Placement.Post) ~flags:L.Flags.all;
  ]

let desc ~precision (c : Case.t) =
  let d = c.Case.width / c.Case.heads in
  let rope =
    match c.Case.rope with
    | Rope_form.No_rope -> None
    | Rope_form.Shared -> Some { L.Rope.batch = 0L; row = i64 d; col = 1L }
    | Rope_form.Batched ->
        Some { L.Rope.batch = i64 (c.Case.tokens * d); row = i64 d; col = 1L }
  and mask =
    if c.Case.mask then
      Some
        {
          A.Strides.batch = i64 (c.Case.tokens * c.Case.tokens);
          head = 0L;
          row = i64 c.Case.tokens;
          col = 1L;
        }
    else None
  in
  L.make ~batch:(i64 c.Case.batch) ~tokens:(i64 c.Case.tokens)
    ~width:(i64 c.Case.width) ~heads:(i64 c.Case.heads) ?norm:c.Case.norm
    ~flags:c.Case.flags ?rope ?mask
    ~scale:(A.Scale.default ~d:(i64 d))
    ~block_keys:(i64 c.Case.block_keys) ~precision ()

let get d = Err.or_raise ~pp_error:L.pp_error d

(* Buffer ids, by role. *)
module Id = struct
  let x = 0
  let norm_weight = 1
  let norm_bias = 2
  let wq = 3
  let bq = 4
  let wk = 5
  let bk = 6
  let wv = 7
  let bv = 8
  let wo = 9
  let bo = 10
  let gamma = 11
  let cos = 12
  let sin = 13
  let mask = 14
  let y = 15
end

let lcg = ref 31337

let uniform ~scale =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  F32_math.round32 (float_of_int ((!lcg / 7 mod 4001) - 2000) /. 2000. *. scale)

let mask_value () =
  lcg := ((!lcg * 1103515245) + 12345) land 0x3FFFFFFF;
  match !lcg / 13 mod 9 with
  | 0 -> Float.neg_infinity
  | n -> float_of_int (n - 4) /. 4.

(* Each operand the layer reads: its id, its [h x w] extents and cells. *)
let operands (t : L.t) (c : Case.t) =
  let rows = c.Case.batch * c.Case.tokens and e = c.Case.width in
  let d = e / c.Case.heads in
  let fill ~h ~w ~scale =
    (h, w, Array.init (h * w) (fun _ -> uniform ~scale))
  in
  let weight_scale = 1. /. sqrt (float_of_int e) in
  let f = t.L.flags in
  let norm_affine =
    match t.L.norm with
    | None -> []
    | Some n ->
        (if n.L.Norm.block.Norm_block.weight then
           [ (Id.norm_weight, fill ~h:1 ~w:e ~scale:1.5) ]
         else [])
        @
        if n.L.Norm.block.Norm_block.bias then
          [ (Id.norm_bias, fill ~h:1 ~w:e ~scale:0.5) ]
        else []
  in
  let optional present id ~scale =
    if present then [ (id, fill ~h:1 ~w:e ~scale) ] else []
  in
  let tables =
    match c.Case.rope with
    | Rope_form.No_rope -> []
    | Rope_form.Shared ->
        [
          (Id.cos, fill ~h:c.Case.tokens ~w:d ~scale:1.);
          (Id.sin, fill ~h:c.Case.tokens ~w:d ~scale:1.);
        ]
    | Rope_form.Batched ->
        [
          (Id.cos, fill ~h:rows ~w:d ~scale:1.);
          (Id.sin, fill ~h:rows ~w:d ~scale:1.);
        ]
  in
  [ (Id.x, fill ~h:rows ~w:e ~scale:2.) ]
  @ norm_affine
  @ [ (Id.wq, fill ~h:e ~w:e ~scale:weight_scale) ]
  @ optional f.L.Flags.q_bias Id.bq ~scale:0.25
  @ [ (Id.wk, fill ~h:e ~w:e ~scale:weight_scale) ]
  @ optional f.L.Flags.k_bias Id.bk ~scale:0.25
  @ [ (Id.wv, fill ~h:e ~w:e ~scale:weight_scale) ]
  @ optional f.L.Flags.v_bias Id.bv ~scale:0.25
  @ [ (Id.wo, fill ~h:e ~w:e ~scale:weight_scale) ]
  @ optional f.L.Flags.o_bias Id.bo ~scale:0.25
  @ optional f.L.Flags.layer_scale Id.gamma ~scale:1.
  @ tables
  @
  if c.Case.mask then
    [
      ( Id.mask,
        ( rows,
          c.Case.tokens,
          Array.init (rows * c.Case.tokens) (fun _ -> mask_value ()) ) );
    ]
  else []

let buffers (t : L.t) operands =
  let rows = Int64.to_int (Int64.mul t.L.batch t.L.tokens)
  and e = Int64.to_int t.L.width in
  List.map
    (fun (id, (h, w, _)) ->
      F.buffer id ~h:(i64 h) ~w:(i64 w) Ssa_format.F32 Ssa_buffer.Input)
    operands
  @ [ F.buffer Id.y ~h:(i64 rows) ~w:(i64 e) Ssa_format.F32 Ssa_buffer.Output ]

let roles operands =
  let has id = List.mem_assoc id operands in
  let opt id = if has id then Some (F.buf id) else None in
  {
    Ssa_attention_layer.Buffers.x = F.buf Id.x;
    norm_weight = opt Id.norm_weight;
    norm_bias = opt Id.norm_bias;
    wq = F.buf Id.wq;
    bq = opt Id.bq;
    wk = F.buf Id.wk;
    bk = opt Id.bk;
    wv = F.buf Id.wv;
    bv = opt Id.bv;
    wo = F.buf Id.wo;
    bo = opt Id.bo;
    gamma = opt Id.gamma;
    cos = opt Id.cos;
    sin = opt Id.sin;
    mask = opt Id.mask;
    y = F.buf Id.y;
  }

let reference (t : L.t) operands =
  let read id o =
    match List.assoc_opt id operands with
    | Some (_, _, cells) -> cells.(Int64.to_int o)
    | None -> Fmt.failwith "the reference read operand %d, which is absent" id
  in
  let out =
    Array.make
      (Int64.to_int (Int64.mul (Int64.mul t.L.batch t.L.tokens) t.L.width))
      0.
  in
  L.eval t
    {
      L.Readers.x = read Id.x;
      norm_weight = read Id.norm_weight;
      norm_bias = read Id.norm_bias;
      wq = read Id.wq;
      bq = read Id.bq;
      wk = read Id.wk;
      bk = read Id.bk;
      wv = read Id.wv;
      bv = read Id.bv;
      wo = read Id.wo;
      bo = read Id.bo;
      gamma = read Id.gamma;
      cos = read Id.cos;
      sin = read Id.sin;
      mask = read Id.mask;
    }
    ~write:(fun o v -> out.(Int64.to_int o) <- v);
  out

(* Everything a backend needs: the expanded, narrowed program, its inputs by
   id with their [h x w] extents, the output's length and the reference. *)
module Prepared = struct
  type t = {
    program : Ssa_program.t;
    inputs : (int * (int * int * float array)) list;
    cells : int;
    width : int;
    want : float array;
  }
end

let prepare ~precision c =
  let t = get (desc ~precision c) in
  let operands = operands t c in
  let p =
    Ssa_attention_layer.program t (roles operands) ~buffers:(buffers t operands)
  in
  {
    Prepared.program =
      Ssa_precision.to_f32 ~exact:true (Ssa_expand_attend.program p);
    inputs = operands;
    cells = c.Case.batch * c.Case.tokens * c.Case.width;
    width = c.Case.width;
    want = reference t operands;
  }

let bitwise a b =
  Array.length a = Array.length b
  && Array.for_all2 Core.Float_bits.equal_portable a b

let interpret (pr : Prepared.t) =
  let memory =
    F.memory
      (List.map
         (fun (id, (_, _, cells)) -> (id, F.floats (Array.copy cells)))
         pr.Prepared.inputs
      @ [ (Id.y, F.floats (Array.make pr.Prepared.cells 0.)) ])
  in
  let memory =
    List.fold_left
      (fun memory (b : Ssa_buffer.t) ->
        match b.Ssa_buffer.role with
        | Ssa_buffer.Scratch ->
            let n = Int64.to_int (Option.get (Ssa_buffer.elements b.extents)) in
            Ssa_id.Buffer.Map.add b.Ssa_buffer.id
              (F.floats (Array.make n 0.))
              memory
        | Ssa_buffer.Input | Ssa_buffer.Output -> memory)
      memory pr.Prepared.program.Ssa_program.buffers
  in
  F.run pr.Prepared.program ~memory;
  match Ssa_id.Buffer.Map.find (F.buf Id.y) memory with
  | Ssa_memory.Floats a -> a
  | _ -> assert false

(* Every case at both binary32 precisions through [run], one line each. *)
let sweep ~backend run =
  List.iter
    (fun precision ->
      List.iter
        (fun (c : Case.t) ->
          let pr = prepare ~precision c in
          let want = pr.Prepared.want in
          let finite =
            Array.fold_left
              (fun n x -> if Float.is_finite x && x <> 0. then n + 1 else n)
              0 want
          in
          Fmt.pr "%s, %s: %s %s (%d of %d nonzero finite)@." (P.name precision)
            c.Case.name backend
            (if bitwise (run pr) want then "equals the reference"
             else "DIFFERS from the reference")
            finite (Array.length want))
        cases)
    [ P.F32_ordered; P.F32_fused ]

let%expect_test "an attention layer's sequence equals the reference" =
  sweep ~backend:"the interpreter" interpret;
  [%expect
    {|
    f32_ordered, bare: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_ordered, pre layer, all flags, masked: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_ordered, post layer, all flags: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_ordered, pre rms, qkv biases, shared rope, masked: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_ordered, pre layer w, all flags, batched rope: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_ordered, no norm, all flags, shared rope, masked, 7 tokens: the interpreter equals the reference (112 of 112 nonzero finite)
    f32_ordered, post rms, one head of 12, 9 tokens: the interpreter equals the reference (216 of 216 nonzero finite)
    f32_fused, bare: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_fused, pre layer, all flags, masked: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_fused, post layer, all flags: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_fused, pre rms, qkv biases, shared rope, masked: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_fused, pre layer w, all flags, batched rope: the interpreter equals the reference (80 of 80 nonzero finite)
    f32_fused, no norm, all flags, shared rope, masked, 7 tokens: the interpreter equals the reference (112 of 112 nonzero finite)
    f32_fused, post rms, one head of 12, 9 tokens: the interpreter equals the reference (216 of 216 nonzero finite) |}]
