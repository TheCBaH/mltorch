(* A granted attention layer as a sequence of existing operations: the norm's
   loops, three projections as [Contract]s with their bias loops, the rotation,
   one [Attend], the output projection and its epilogue, and a Post norm. No
   operation of its own: every backend that realizes a [Contract] and expands a
   binary32 [Attend] computes it. *)

module B = Ssa_builder
module L = Attention_layer_block
module P = Attention_block.Precision
open Ssa_expand_util

module Buffers = struct
  type t = {
    x : Ssa_id.Buffer.t;
    norm_weight : Ssa_id.Buffer.t option;
    norm_bias : Ssa_id.Buffer.t option;
    wq : Ssa_id.Buffer.t;
    bq : Ssa_id.Buffer.t option;
    wk : Ssa_id.Buffer.t;
    bk : Ssa_id.Buffer.t option;
    wv : Ssa_id.Buffer.t;
    bv : Ssa_id.Buffer.t option;
    wo : Ssa_id.Buffer.t;
    bo : Ssa_id.Buffer.t option;
    gamma : Ssa_id.Buffer.t option;
    cos : Ssa_id.Buffer.t option;
    sin : Ssa_id.Buffer.t option;
    mask : Ssa_id.Buffer.t option;
    y : Ssa_id.Buffer.t;
  }
end

let flat ?(format = Ssa_format.F32) id n =
  {
    Ssa_buffer.id = Ssa_id.Buffer.of_int id;
    extents =
      Expr.Coord.of_fn (fun a ->
          match a with
          | Expr.Axis.C -> n
          | Expr.Axis.D | Expr.Axis.H | Expr.Axis.N | Expr.Axis.T | Expr.Axis.W
            ->
              1L);
    format;
    role = Ssa_buffer.Scratch;
  }

let placement (t : L.t) =
  Option.map (fun (n : L.Norm.t) -> n.L.Norm.placement) t.L.norm

let bf16 (t : L.t) = t.L.precision = P.Bf16_paired

(* The scratch, in a fixed order: u (Pre, or every bfloat16 layer: the paired
   products' bfloat16 operand), q, k, v, a, o, r (Post only), then for a
   bfloat16 layer the bfloat16 copies of q, k, v and a the attention and the
   output product read. *)
let scratch (t : L.t) ~first =
  let cells = Int64.mul (Int64.mul t.L.batch t.L.tokens) t.L.width in
  let pre = placement t = Some L.Placement.Pre || bf16 t
  and post = placement t = Some L.Placement.Post in
  let first = (first : Ssa_id.Buffer.t :> int) in
  let u_format = if bf16 t then Ssa_format.Bf16 else Ssa_format.F32 in
  let f32 = 5 + Bool.to_int post in
  (if pre then [ flat ~format:u_format first cells ] else [])
  @ List.init f32 (fun i -> flat (first + Bool.to_int pre + i) cells)
  @
  if bf16 t then
    List.init 4 (fun i ->
        flat ~format:Ssa_format.Bf16 (first + Bool.to_int pre + f32 + i) cells)
  else []

let need name = function
  | Some b -> b
  | None -> invalid_arg ("Ssa_attention_layer: no " ^ name ^ " buffer")

let body b (t : L.t) (bf : Buffers.t) ~scratch =
  let ids = List.map id scratch in
  let pre = placement t = Some L.Placement.Pre || bf16 t
  and post = placement t = Some L.Placement.Post in
  let take = ref ids in
  let next () =
    match !take with
    | x :: rest ->
        take := rest;
        x
    | [] -> invalid_arg "Ssa_attention_layer: too little scratch"
  in
  let u = if pre then Some (next ()) else None in
  let q = next () in
  let k = next () in
  let v = next () in
  let a = next () in
  let o = next () in
  let r = if post then Some (next ()) else None in
  let qb, kb, vb, ab =
    if bf16 t then
      let qb = next () in
      let kb = next () in
      let vb = next () in
      let ab = next () in
      (qb, kb, vb, ab)
    else (q, k, v, a)
  in
  let f = t.L.flags in
  let opt present name buffer =
    if present then Some (need name buffer) else None
  in
  let norm_operands (n : Norm_block.t) =
    ( opt n.Norm_block.weight "norm weight" bf.Buffers.norm_weight,
      opt n.Norm_block.bias "norm bias" bf.Buffers.norm_bias )
  in
  let e = t.L.width and rows = Int64.mul t.L.batch t.L.tokens in
  let binary b op x y = B.f64_binary b op x y in
  (* every cell [i * e + c] of a [rows][e] buffer *)
  let cells body =
    loop b rows (fun b i ->
        loop b e (fun b c -> body b (offset b [ (e, i); (1L, c) ]) c))
  in
  (* a cell-for-cell copy rounded to bfloat16 *)
  let to_bf16 ~src ~dst =
    cells (fun b at _ -> store_bf16 b dst at (load b src at))
  in
  let src =
    match (t.L.norm, u) with
    | Some n, Some u when n.L.Norm.placement = L.Placement.Pre ->
        let weight, bias = norm_operands n.L.Norm.block in
        Ssa_norm.body b n.L.Norm.block ~x:bf.Buffers.x ~weight ~bias ~y:u;
        u
    | _, Some u ->
        (* a bfloat16 layer without a Pre norm: x itself is the operand *)
        to_bf16 ~src:bf.Buffers.x ~dst:u;
        u
    | _, None -> bf.Buffers.x
  in
  let project desc ~a:src ~w ~c:out ~bias =
    B.contract b desc ~a:src ~b:w ~c:out;
    match bias with
    | None -> ()
    | Some bias ->
        cells (fun b at c ->
            store b out at
              (binary b Expr.Value.Add (load b out at) (load b bias c)))
  in
  project t.L.q ~a:src ~w:bf.Buffers.wq ~c:q
    ~bias:(opt f.L.Flags.q_bias "q bias" bf.Buffers.bq);
  project t.L.k ~a:src ~w:bf.Buffers.wk ~c:k
    ~bias:(opt f.L.Flags.k_bias "k bias" bf.Buffers.bk);
  project t.L.v ~a:src ~w:bf.Buffers.wv ~c:v
    ~bias:(opt f.L.Flags.v_bias "v bias" bf.Buffers.bv);
  (match t.L.rope with
  | None -> ()
  | Some rs ->
      let cos = need "cos" bf.Buffers.cos and sin = need "sin" bf.Buffers.sin in
      let d = t.L.d in
      let half = Int64.div d 2L in
      let minus_one = B.f64 b (-1.) in
      let rotate z =
        loop b t.L.batch (fun b bb ->
            loop b t.L.tokens (fun b tok ->
                loop b t.L.heads (fun b h ->
                    loop b half (fun b ff ->
                        let lo =
                          offset b
                            [
                              (Int64.mul t.L.tokens e, bb);
                              (e, tok);
                              (d, h);
                              (1L, ff);
                            ]
                        in
                        let hi = B.index_add b lo (B.index b half) in
                        let table buf ~upper =
                          let feature =
                            if upper then B.index_add b ff (B.index b half)
                            else ff
                          in
                          load b buf
                            (offset b
                               [
                                 (rs.L.Rope.batch, bb);
                                 (rs.L.Rope.row, tok);
                                 (rs.L.Rope.col, feature);
                               ])
                        in
                        let x0 = load b z lo and x1 = load b z hi in
                        let rotated x partner ~upper =
                          binary b Expr.Value.Add
                            (binary b Expr.Value.Mul x (table cos ~upper))
                            (binary b Expr.Value.Mul partner (table sin ~upper))
                        in
                        let y0 =
                          rotated x0
                            (binary b Expr.Value.Mul x1 minus_one)
                            ~upper:false
                        and y1 = rotated x1 x0 ~upper:true in
                        store b z lo y0;
                        store b z hi y1))))
      in
      rotate q;
      rotate k);
  if bf16 t then (
    to_bf16 ~src:q ~dst:qb;
    to_bf16 ~src:k ~dst:kb;
    to_bf16 ~src:v ~dst:vb);
  B.attend b t.L.attend ~q:qb ~k:kb ~v:vb ~mask:bf.Buffers.mask ~o:a;
  if bf16 t then to_bf16 ~src:a ~dst:ab;
  B.contract b t.L.o ~a:ab ~b:bf.Buffers.wo ~c:o;
  let bo = opt f.L.Flags.o_bias "o bias" bf.Buffers.bo
  and gamma = opt f.L.Flags.layer_scale "gamma" bf.Buffers.gamma in
  let dst = match r with Some r -> r | None -> bf.Buffers.y in
  cells (fun b at c ->
      let x = load b o at in
      let x =
        match bo with
        | Some bo -> binary b Expr.Value.Add x (load b bo c)
        | None -> x
      in
      let x =
        match gamma with
        | Some g -> binary b Expr.Value.Mul x (load b g c)
        | None -> x
      in
      store b dst at (binary b Expr.Value.Add x (load b bf.Buffers.x at)));
  match (t.L.norm, r) with
  | Some n, Some r ->
      let weight, bias = norm_operands n.L.Norm.block in
      Ssa_norm.body b n.L.Norm.block ~x:r ~weight ~bias ~y:bf.Buffers.y
  | _ -> ()

let program t bf ~buffers =
  let first =
    1
    + List.fold_left
        (fun m (x : Ssa_buffer.t) -> max m (x.Ssa_buffer.id :> int))
        0 buffers
  in
  let scratch = scratch t ~first:(Ssa_id.Buffer.of_int first) in
  template ~buffers:(buffers @ scratch) (fun b -> body b t bf ~scratch)
