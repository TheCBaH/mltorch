(* A binary32 attention as ordinary operations. Each [Attend] whose precision is
   binary32 becomes the loops of {!Attention_block.eval}: per batch and head,
   the keys in blocks, each block's scores and context as one [Contract] over
   dense scratch copies of Q, K and V (a [Contract] addresses whole buffers, so
   the blocks are staged), and the running maximum, sum and accumulator updated
   per query in between. No emitter sees an [Attend] afterwards. A bfloat16
   attention is left alone: it is the matrix engine's, and only the Machine IR
   route realizes it.

   The expansion is written once, with the builder, as a small program over the
   attention's own buffers plus its scratch; its body is then cloned in place of
   the [Attend] and the [Attend]'s effect redirected to the clone's last. The
   arithmetic is binary64 until {!Ssa_precision.to_f32}, as every lowered
   program's is. *)

module A = Attention_block
module B = Ssa_builder

let scratch_buffer id n =
  {
    Ssa_buffer.id = Ssa_id.Buffer.of_int id;
    extents =
      Expr.Coord.of_fn (fun a ->
          match a with
          | Expr.Axis.C -> n
          | Expr.Axis.D | Expr.Axis.H | Expr.Axis.N | Expr.Axis.T | Expr.Axis.W
            ->
              1L);
    format = Ssa_format.F32;
    role = Ssa_buffer.Scratch;
  }

(* The scratch an attention needs: Q, a block of K and of V, the block's scores
   and probabilities, the accumulator, and the running maximum and sum. *)
module Scratch = struct
  type t = {
    acc : Ssa_buffer.t;
    k : Ssa_buffer.t;
    l : Ssa_buffer.t;
    m : Ssa_buffer.t;
    p : Ssa_buffer.t;
    q : Ssa_buffer.t;
    s : Ssa_buffer.t;
    v : Ssa_buffer.t;
  }

  let make ~first (d : A.t) =
    let ( * ) = Int64.mul in
    let next = ref first in
    let one n =
      let b = scratch_buffer !next n in
      incr next;
      b
    in
    let acc = one (d.A.tq * d.A.d) in
    let k = one (d.A.block_keys * d.A.d) in
    let l = one d.A.tq in
    let m = one d.A.tq in
    let p = one (d.A.tq * d.A.block_keys) in
    let q = one (d.A.tq * d.A.d) in
    let s = one (d.A.block_keys * d.A.tq) in
    let v = one (d.A.block_keys * d.A.d) in
    { acc; k; l; m; p; q; s; v }

  let all t = [ t.acc; t.k; t.l; t.m; t.p; t.q; t.s; t.v ]
  let count = 8
end

let contraction ~batch ~m ~n ~k ~a ~b ~c ~seed (d : A.t) =
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch ~m ~n ~k ~a ~b ~c
          ~format:(A.Precision.format d.A.precision)
          ~seed
          ~numerics:(A.Precision.numerics d.A.precision)))

open Ssa_expand_util

(* The program of one attention: the same buffers, the scratch, and the
   recurrence. *)
let expansion (d : A.t) ~q ~k ~v ~mask ~o ~(scratch : Scratch.t) ~buffers =
  let ( * ) = Int64.mul in
  let sc = scratch in
  let build b =
    let f x = B.f64 b x in
    let tq = d.A.tq and dd = d.A.d and bk = d.A.block_keys in
    let nfull = Int64.div d.A.tk bk and rem = Int64.rem d.A.tk bk in
    let binary b op x y = B.f64_binary b op x y in
    let ninf = f Float.neg_infinity and zero = f 0. and one = f 1. in
    let scale = f d.A.scale in
    (* rows [r < rows] of [src] from [base + (row0 + r) * row + f * col] into
       the dense [rows x d] scratch [dst] *)
    let stage b ~src ~dst ~rows ~base ~row0 ~(s : A.Strides.t) =
      loop b rows (fun b r ->
          loop b dd (fun b feature ->
              let at =
                offset b
                  [
                    (1L, base);
                    (s.A.Strides.row, row0);
                    (s.A.Strides.row, r);
                    (s.A.Strides.col, feature);
                  ]
              in
              store b dst (offset b [ (dd, r); (1L, feature) ]) (load b src at)))
    in
    loop b d.A.batch (fun b bb ->
        loop b d.A.heads (fun b hh ->
            let base (s : A.Strides.t) =
              offset b [ (s.A.Strides.batch, bb); (s.A.Strides.head, hh) ]
            in
            let zero_i = B.index b 0L in
            stage b ~src:q ~dst:(id sc.Scratch.q) ~rows:tq ~base:(base d.A.q)
              ~row0:zero_i ~s:d.A.q;
            loop b tq (fun b i ->
                store b (id sc.Scratch.m) i ninf;
                store b (id sc.Scratch.l) i zero;
                loop b dd (fun b feature ->
                    store b (id sc.Scratch.acc)
                      (offset b [ (dd, i); (1L, feature) ])
                      zero));
            let block b ~k0 ~vk =
              let vki = Int64.of_int vk in
              stage b ~src:k ~dst:(id sc.Scratch.k) ~rows:vki ~base:(base d.A.k)
                ~row0:k0 ~s:d.A.k;
              stage b ~src:v ~dst:(id sc.Scratch.v) ~rows:vki ~base:(base d.A.v)
                ~row0:k0 ~s:d.A.v;
              (* scores [vk][tq] = K Q^T *)
              B.contract b
                (contraction ~batch:1L ~m:vki ~n:tq ~k:dd
                   ~a:{ Contraction.Operand.batch = 0L; outer = dd; k = 1L }
                   ~b:{ Contraction.Operand.batch = 0L; outer = dd; k = 1L }
                   ~c:{ Contraction.Output.batch = 0L; row = tq; col = 1L }
                   ~seed:Contraction.Seed.Zero d)
                ~a:(id sc.Scratch.k) ~b:(id sc.Scratch.q) ~c:(id sc.Scratch.s);
              loop b tq (fun b i ->
                  let z b r =
                    let s =
                      load b (id sc.Scratch.s) (offset b [ (tq, r); (1L, i) ])
                    in
                    let scaled = binary b Expr.Value.Mul s scale in
                    match (d.A.mask, mask) with
                    | Some ms, Some mask ->
                        let at =
                          offset b
                            [
                              (ms.A.Strides.batch, bb);
                              (ms.A.Strides.head, hh);
                              (ms.A.Strides.row, i);
                              (ms.A.Strides.col, k0);
                              (ms.A.Strides.col, r);
                            ]
                        in
                        binary b Expr.Value.Add scaled (load b mask at)
                    | _ -> scaled
                  in
                  let m_old = load b (id sc.Scratch.m) i in
                  let hi = B.index b vki in
                  let m_new =
                    match
                      B.for_ b ~lo:(B.index b 0L) ~hi
                        ~init:(B.Cons (m_old, B.Nil))
                        (fun b r (B.Cons (m, B.Nil)) ->
                          B.Cons (B.f64_max b m (z b r), B.Nil))
                    with
                    | B.Cons (m, B.Nil) -> m
                  in
                  let empty = B.float_compare b Ssa_op.Compare.Eq m_new ninf in
                  let a =
                    B.select b empty one
                      (B.f64_unary b Expr.Value.Exp
                         (binary b Expr.Value.Sub m_old m_new))
                  in
                  let sum =
                    match
                      B.for_ b ~lo:(B.index b 0L) ~hi
                        ~init:(B.Cons (zero, B.Nil))
                        (fun b r (B.Cons (sum, B.Nil)) ->
                          let zr = z b r in
                          let gone =
                            B.float_compare b Ssa_op.Compare.Eq zr ninf
                          in
                          let p =
                            B.select b gone zero
                              (B.f64_unary b Expr.Value.Exp
                                 (binary b Expr.Value.Sub zr m_new))
                          in
                          store b (id sc.Scratch.p)
                            (offset b [ (bk, i); (1L, r) ])
                            p;
                          B.Cons (binary b Expr.Value.Add sum p, B.Nil))
                    with
                    | B.Cons (sum, B.Nil) -> sum
                  in
                  store b (id sc.Scratch.m) i m_new;
                  let l_old = load b (id sc.Scratch.l) i in
                  store b (id sc.Scratch.l) i
                    (binary b Expr.Value.Add
                       (binary b Expr.Value.Mul l_old a)
                       sum);
                  loop b dd (fun b feature ->
                      let at = offset b [ (dd, i); (1L, feature) ] in
                      store b (id sc.Scratch.acc) at
                        (binary b Expr.Value.Mul a
                           (load b (id sc.Scratch.acc) at))));
              (* context: the accumulator, scaled above, plus P V *)
              B.contract b
                (contraction ~batch:1L ~m:tq ~n:dd ~k:vki
                   ~a:{ Contraction.Operand.batch = 0L; outer = bk; k = 1L }
                   ~b:{ Contraction.Operand.batch = 0L; outer = 1L; k = dd }
                   ~c:{ Contraction.Output.batch = 0L; row = dd; col = 1L }
                   ~seed:Contraction.Seed.Tensor d)
                ~a:(id sc.Scratch.p) ~b:(id sc.Scratch.v) ~c:(id sc.Scratch.acc)
            in
            loop b nfull (fun b j ->
                block b ~k0:(B.index_scale b bk j) ~vk:(Int64.to_int bk));
            if Int64.compare rem 0L > 0 then
              block b ~k0:(B.index b (nfull * bk)) ~vk:(Int64.to_int rem);
            let o_base = base d.A.o in
            loop b tq (fun b i ->
                let l = load b (id sc.Scratch.l) i in
                let none = B.float_compare b Ssa_op.Compare.Eq l zero in
                loop b dd (fun b feature ->
                    let acc =
                      load b (id sc.Scratch.acc)
                        (offset b [ (dd, i); (1L, feature) ])
                    in
                    let out =
                      B.select b none zero (binary b Expr.Value.Div acc l)
                    in
                    store b o
                      (offset b
                         [
                           (1L, o_base);
                           (d.A.o.A.Strides.row, i);
                           (d.A.o.A.Strides.col, feature);
                         ])
                      out))))
  in
  template ~buffers build

let program (p : Ssa_program.t) =
  let next_buffer =
    ref
      (1
      + List.fold_left
          (fun m (b : Ssa_buffer.t) -> max m (b.Ssa_buffer.id :> int))
          0 p.Ssa_program.buffers)
  in
  let added = ref [] in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr
        {
          Ssa_instr.op = Ssa_op.Attend { desc; q; k; v; mask; o };
          token = Some token;
          results = [ after ];
          _;
        }
      when desc.A.precision <> A.Precision.Bf16_paired ->
        let scratch = Scratch.make ~first:!next_buffer desc in
        next_buffer := !next_buffer + Scratch.count;
        added := !added @ Scratch.all scratch;
        let declared = declared p ([ q; k; v; o ] @ Option.to_list mask) in
        let template =
          expansion desc ~q ~k ~v ~mask ~o ~scratch
            ~buffers:(declared @ Scratch.all scratch)
        in
        splice t template ~token ~after
    | _ -> [ s ]
  in
  let p', changed = Ssa_rewrite.program rule p in
  if changed then
    { p' with Ssa_program.buffers = p'.Ssa_program.buffers @ !added }
  else p
