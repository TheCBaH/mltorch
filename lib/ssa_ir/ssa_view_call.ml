module At = struct
  type t = Base of Ssa_value.t | Rows of Ssa_value.t list
end

module View = struct
  type t = { buffer : Ssa_id.Buffer.t; at : At.t }
end

type t = { desc : Contraction.t; a : View.t; b : View.t; c : View.t }

let bases (v : View.t) =
  match v.View.at with At.Base b -> [ b ] | At.Rows rs -> rs

let operands (t : t) = List.concat_map bases [ t.a; t.b; t.c ]

let map_operands f (t : t) =
  let view (v : View.t) =
    let at =
      match v.View.at with
      | At.Base b -> At.Base (f b)
      | At.Rows rs -> At.Rows (List.map f rs)
    in
    { v with View.at }
  in
  let a = view t.a in
  let b = view t.b in
  let c = view t.c in
  { t with a; b; c }

let reads (t : t) = [ t.a.View.buffer; t.b.View.buffer ]
let writes (t : t) = t.c.View.buffer

module Reach = struct
  type t = { a : int64; b : int64; c : int64 }
end

let last count stride = Int64.mul (Int64.max 0L (Int64.pred count)) stride
let sum = List.fold_left Int64.add 0L

(* The last cell from a base. With one base per row the outer stride is not
   part of the offset. *)
let reach (t : t) : Reach.t =
  let d = t.desc in
  let ka = Int64.max 1L d.Contraction.k in
  let operand (v : View.t) (o : Contraction.Operand.t) ~outer =
    let outer_term =
      match v.View.at with
      | At.Base _ -> last outer o.Contraction.Operand.outer
      | At.Rows _ -> 0L
    in
    sum
      [
        last d.Contraction.batch o.Contraction.Operand.batch;
        outer_term;
        last ka o.Contraction.Operand.k;
      ]
  in
  let co = d.Contraction.c in
  let c_outer =
    match t.c.View.at with
    | At.Base _ -> last d.Contraction.m co.Contraction.Output.row
    | At.Rows _ -> 0L
  in
  {
    a = operand t.a d.Contraction.a ~outer:d.Contraction.m;
    b = operand t.b d.Contraction.b ~outer:d.Contraction.n;
    c =
      sum
        [
          last d.Contraction.batch co.Contraction.Output.batch;
          c_outer;
          last d.Contraction.n co.Contraction.Output.col;
        ];
  }

let rows_expected (t : t) =
  let d = t.desc in
  let count v n =
    match v.View.at with
    | At.Base _ -> true
    | At.Rows rs -> Int64.equal (Int64.of_int (List.length rs)) n
  in
  count t.a d.Contraction.m && count t.b d.Contraction.n
  && count t.c d.Contraction.m

(* A descriptor with the same extents, format and numerics whose offsets decode
   to (batch, outer, inner) without ambiguity: inner stride one, outer stride
   the inner extent, batch stride the whole plane. *)
let virtual_desc (d : Contraction.t) =
  let k = Int64.max 1L d.Contraction.k in
  let n = d.Contraction.n and m = d.Contraction.m in
  Result.get_ok
    (Err.payload
       (Contraction.make ~batch:d.Contraction.batch ~m ~n ~k:d.Contraction.k
          ~a:{ Contraction.Operand.batch = Int64.mul m k; outer = k; k = 1L }
          ~b:{ Contraction.Operand.batch = Int64.mul n k; outer = k; k = 1L }
          ~c:{ Contraction.Output.batch = Int64.mul m n; row = n; col = 1L }
          ~format:d.Contraction.format ~seed:d.Contraction.seed
          ~numerics:d.Contraction.numerics))

let eval (t : t) ~base ~read ~write =
  let d = t.desc in
  let v = virtual_desc d in
  let k = Int64.max 1L d.Contraction.k in
  let cell (view : View.t) ~batch_stride ~outer_stride ~inner_stride ~outer_n
      ~inner_n off =
    let plane = Int64.mul outer_n inner_n in
    let b = Int64.div off plane in
    let r = Int64.rem off plane in
    let o = Int64.div r inner_n and x = Int64.rem r inner_n in
    let tail =
      Int64.add (Int64.mul b batch_stride) (Int64.mul x inner_stride)
    in
    match view.View.at with
    | At.Base v ->
        Int64.add (base v) (Int64.add tail (Int64.mul o outer_stride))
    | At.Rows rs -> Int64.add (base (List.nth rs (Int64.to_int o))) tail
  in
  let operand (view : View.t) (o : Contraction.Operand.t) ~outer_n =
    cell view ~batch_stride:o.Contraction.Operand.batch
      ~outer_stride:o.Contraction.Operand.outer
      ~inner_stride:o.Contraction.Operand.k ~outer_n ~inner_n:k
  in
  let co = d.Contraction.c in
  let c_cell =
    cell t.c ~batch_stride:co.Contraction.Output.batch
      ~outer_stride:co.Contraction.Output.row
      ~inner_stride:co.Contraction.Output.col ~outer_n:d.Contraction.m
      ~inner_n:d.Contraction.n
  in
  let a_cell = operand t.a d.Contraction.a ~outer_n:d.Contraction.m in
  let b_cell = operand t.b d.Contraction.b ~outer_n:d.Contraction.n in
  Contraction.eval v
    ~a:(fun off -> read t.a.View.buffer (a_cell off))
    ~b:(fun off -> read t.b.View.buffer (b_cell off))
    ~c:(fun off -> read t.c.View.buffer (c_cell off))
    ~write:(fun off x -> write (c_cell off) x)
