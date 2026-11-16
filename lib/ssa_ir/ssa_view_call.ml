module View = struct
  type t = { buffer : Ssa_id.Buffer.t; base : Ssa_value.t }
end

type t = { desc : Contraction.t; a : View.t; b : View.t; c : View.t }

let operands t = [ t.a.View.base; t.b.View.base; t.c.View.base ]

let map_operands f t =
  let view (v : View.t) = { v with View.base = f v.View.base } in
  let a = view t.a in
  let b = view t.b in
  let c = view t.c in
  { t with a; b; c }

let reads t = [ t.a.View.buffer; t.b.View.buffer ]
let writes t = t.c.View.buffer

module Reach = struct
  type t = { a : int64; b : int64; c : int64 }
end

let reach (d : Contraction.t) : Reach.t =
  let last count stride = Int64.mul (Int64.max 0L (Int64.pred count)) stride in
  let ka = Int64.max 1L d.Contraction.k in
  let sum = List.fold_left Int64.add 0L in
  let operand (o : Contraction.Operand.t) ~outer =
    sum
      [
        last d.Contraction.batch o.Contraction.Operand.batch;
        last outer o.Contraction.Operand.outer;
        last ka o.Contraction.Operand.k;
      ]
  in
  {
    a = operand d.Contraction.a ~outer:d.Contraction.m;
    b = operand d.Contraction.b ~outer:d.Contraction.n;
    c =
      sum
        [
          last d.Contraction.batch d.Contraction.c.Contraction.Output.batch;
          last d.Contraction.m d.Contraction.c.Contraction.Output.row;
          last d.Contraction.n d.Contraction.c.Contraction.Output.col;
        ];
  }

let eval (t : t) ~base ~read ~write =
  let shifted (v : View.t) off =
    read v.View.buffer (Int64.add (base v.View.base) off)
  in
  let origin = base t.c.View.base in
  Contraction.eval t.desc ~a:(shifted t.a) ~b:(shifted t.b) ~c:(shifted t.c)
    ~write:(fun off x -> write (Int64.add origin off) x)
