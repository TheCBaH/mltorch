(* See coord.mli. *)

type 'a t = { n : 'a; t : 'a; d : 'a; h : 'a; w : 'a; c : 'a }

let make ~n ~t ~d ~h ~w ~c = { n; t; d; h; w; c }

let of_fn f =
  {
    n = f Axis.N;
    t = f Axis.T;
    d = f Axis.D;
    h = f Axis.H;
    w = f Axis.W;
    c = f Axis.C;
  }

let get v : Axis.t -> 'a = function
  | N -> v.n
  | T -> v.t
  | D -> v.d
  | H -> v.h
  | W -> v.w
  | C -> v.c

let set v (a : Axis.t) x =
  match a with
  | N -> { v with n = x }
  | T -> { v with t = x }
  | D -> { v with d = x }
  | H -> { v with h = x }
  | W -> { v with w = x }
  | C -> { v with c = x }

let map f v =
  { n = f v.n; t = f v.t; d = f v.d; h = f v.h; w = f v.w; c = f v.c }

let mapi f v =
  {
    n = f Axis.N v.n;
    t = f Axis.T v.t;
    d = f Axis.D v.d;
    h = f Axis.H v.h;
    w = f Axis.W v.w;
    c = f Axis.C v.c;
  }

let fold f acc v = f (f (f (f (f (f acc v.n) v.t) v.d) v.h) v.w) v.c

let foldi f acc v =
  let acc = f Axis.N acc v.n in
  let acc = f Axis.T acc v.t in
  let acc = f Axis.D acc v.d in
  let acc = f Axis.H acc v.h in
  let acc = f Axis.W acc v.w in
  f Axis.C acc v.c

let for_all p v = fold (fun acc x -> acc && p x) true v
let to_list v = List.map (get v) Axis.all

(* [Fmt.any ","], not [Fmt.comma]: the latter is a breakable ",@ " and this
   separator must never break -- the printed form is pinned by expect tests. *)
let pp pp_a fmt v = Fmt.list ~sep:(Fmt.any ",") pp_a fmt (to_list v)
