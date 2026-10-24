(* Matrix products in a structured binary32 program: an ordered sum from 0 to a
   constant K, seeded with positive zero, whose body marks a reduction and
   multiplies one in-bounds load of A by one of B, each indexed by constant
   zeros, the sum's own variable and the variables of the loops around it
   (each from 0 by 1 to a constant). The sum's variable must index both, once;
   a loop variable indexing only A is the row, only B the column, both the
   batch. The strides follow from each buffer's row-major extents.

   Every ordered sum is either found or refused with a reason. *)

module Refusal = struct
  type t =
    | Batch_loops  (** more than one loop indexes both operands *)
    | Binary64  (** the sum is not binary32 *)
    | Bounds  (** a range not from constant 0 by 1 to a constant *)
    | Format  (** the operands' loads are not both binary32 or both bfloat16 *)
    | Guarded  (** a checked access inside the reduction *)
    | Non_affine
        (** a coordinate neither zero nor a loop variable, or a variable on two
            axes of one operand *)
    | Not_product  (** the term is not one product of two loads *)
    | Operand_written  (** A or B is stored inside the loops around the sum *)
    | Seed  (** the sum starts from something other than positive zero *)
    | Shape
        (** the sum's variable not on exactly one axis of each operand, or more
            than one row or column loop *)

  let name = function
    | Batch_loops -> "batch_loops"
    | Binary64 -> "binary64"
    | Bounds -> "bounds"
    | Format -> "format"
    | Guarded -> "guarded"
    | Non_affine -> "non_affine"
    | Not_product -> "not_product"
    | Operand_written -> "operand_written"
    | Seed -> "seed"
    | Shape -> "shape"
end

(* A recognized product: its descriptor, its operand buffers, and the loop
   variables it is indexed by (batch, row, column), as the rewrite needs. *)
module Found = struct
  type t = {
    desc : Contraction.t;
    a : Ssa_id.Buffer.t;
    b : Ssa_id.Buffer.t;
    sum : Ssa_value.t;  (** the sum's result *)
    batch : Ssa_value.t option;
    rows : (Ssa_value.t * int64) list;
        (** the row loops, outer first, with their extents: none, one, or two
            that flatten into one *)
    col : Ssa_value.t option;
    top : Ssa_value.t option;
        (** the induction value of the outermost loop the product's variables
            belong to: the contraction goes just before it ([None]: before the
            sum itself) *)
  }
end

module Outcome = struct
  type t = Found of Found.t | Refused of Refusal.t
end

exception Refuse of Refusal.t

let refuse r = raise (Refuse r)
let key (v : Ssa_value.t) = (v.Ssa_value.id :> int)

(* Every constant of the program, by the value it defines. *)
let constants (p : Ssa_program.t) =
  let t = Hashtbl.create 64 in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt = function
    | Ssa_stmt.Instr { Ssa_instr.op = Ssa_op.Const c; results = [ v ]; _ } ->
        Hashtbl.replace t (key v) c
    | Ssa_stmt.Instr _ -> ()
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } -> region body
    | Ssa_stmt.If { then_; else_; _ } ->
        region then_;
        region else_
  in
  region p.Ssa_program.entry;
  t

(* A loop the sum sits in: its induction value and constant extent. *)
type loop = { iv : Ssa_value.t; extent : int64 option }

let index_const consts (v : Ssa_value.t) =
  match Hashtbl.find_opt consts (key v) with
  | Some (Ssa_const.Index k) -> Some k
  | _ -> None

(* The buffers a statement list stores to, nested regions included. *)
let rec stored (stmts : Ssa_region.t Ssa_stmt.t list) =
  List.concat_map
    (function
      | Ssa_stmt.Instr
          {
            Ssa_instr.op =
              Ssa_op.Store { buffer; _ } | Ssa_op.Vec_store { buffer; _ };
            _;
          } ->
          [ buffer ]
      | Ssa_stmt.Instr _ -> []
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          stored body.Ssa_region.body
      | Ssa_stmt.If { then_; else_; _ } ->
          stored then_.Ssa_region.body @ stored else_.Ssa_region.body)
    stmts

(* Row-major element strides of six extents. *)
let strides (e : int64 Expr.Coord.t) =
  let es = Expr.Coord.to_list e in
  let rec go = function
    | [] -> []
    | _ :: rest -> List.fold_left Int64.mul 1L rest :: go rest
  in
  go es

let recognize ~numerics ~consts ~ranges ~buffer ~(loops : loop list)
    ~nest_stores (lo, hi, seed, results, (body : Ssa_region.t)) =
  (* a value the program or its range analysis fixes *)
  let known (v : Ssa_value.t) =
    match index_const consts v with
    | Some k -> Some k
    | None -> (
        match Ssa_range.range ranges v with
        | Ssa_range.Range { lo; hi } when Int64.equal lo hi -> Some lo
        | Ssa_range.Range _ | Ssa_range.Empty -> None)
  in
  let positive_zero (v : Ssa_value.t) =
    match Hashtbl.find_opt consts (key v) with
    | Some (Ssa_const.F32 x) -> x = 0. && 1. /. x > 0.
    | _ -> false
  in
  let sum =
    match results with [ s; _ ] -> s | _ -> refuse Refusal.Not_product
  in
  (match sum.Ssa_value.ty with
  | Ssa_type.Scalar Ssa_type.F32 -> ()
  | _ -> refuse Refusal.Binary64);
  (match known lo with Some 0L -> () | _ -> refuse Refusal.Bounds);
  let k = match known hi with Some k -> k | None -> refuse Refusal.Bounds in
  if not (positive_zero seed) then refuse Refusal.Seed;
  let kvar =
    match body.Ssa_region.params with
    | iv :: _ -> iv
    | [] -> refuse Refusal.Shape
  in
  let yielded (r : Ssa_region.t) =
    match r.Ssa_region.yields with
    | [ t; _ ] -> t
    | _ -> refuse Refusal.Not_product
  in
  (* the body: marks, index additions of a proven zero (which stand for their
     other operand), two binary32 or two widened bfloat16 loads and their
     product; or a sum of one
     term over [0, 1) from +0 whose body is that again (a 1x1 window), which
     is the product rounded once and makes the whole an ordered sum *)
  let alias = Hashtbl.create 8 in
  let rec resolve (v : Ssa_value.t) =
    match Hashtbl.find_opt alias (key v) with Some w -> resolve w | None -> v
  in
  let zero v = known (resolve v) = Some 0L in
  let loads = Hashtbl.create 4 and product = ref None and nested = ref false in
  let rec walk (r : Ssa_region.t) term =
    List.iter
      (function
        | Ssa_stmt.Instr { Ssa_instr.op = Ssa_op.Mark Ssa_mark.Reduction; _ } ->
            ()
        | Ssa_stmt.Instr
            {
              Ssa_instr.op = Ssa_op.Index_add_in_domain (a, b);
              results = [ v ];
              _;
            }
          when zero b || zero a ->
            Hashtbl.replace alias (key v) (if zero b then a else b)
        | Ssa_stmt.Instr
            {
              Ssa_instr.op =
                Ssa_op.Load_in_bounds
                  {
                    buffer = id;
                    at;
                    decode =
                      (Ssa_op.Decode.F32_exact | Ssa_op.Decode.Bf16_to_f32) as
                      decode;
                  };
              results = [ v; _ ];
              _;
            } ->
            Hashtbl.replace loads (key v) (id, at, decode)
        | Ssa_stmt.Instr
            { Ssa_instr.op = Ssa_op.Load _ | Ssa_op.Check_access _; _ } ->
            refuse Refusal.Guarded
        | Ssa_stmt.Instr
            {
              Ssa_instr.op = Ssa_op.Float_binary (Expr.Value.Mul, x, y);
              results = [ r ];
              _;
            }
          when key r = key term ->
            product := Some (x, y)
        | Ssa_stmt.Ordered_sum
            { lo; hi; seed; results = s :: _; body = inner; _ }
          when key s = key term ->
            if
              not
                (known lo = Some 0L && known hi = Some 1L && positive_zero seed)
            then refuse Refusal.Bounds;
            nested := true;
            walk inner (yielded inner)
        | _ -> refuse Refusal.Not_product)
      r.Ssa_region.body
  in
  walk body (yielded body);
  let x, y =
    match !product with Some p -> p | None -> refuse Refusal.Not_product
  in
  let load v =
    match Hashtbl.find_opt loads (key v) with
    | Some l -> l
    | None -> refuse Refusal.Not_product
  in
  let (ida, ata, da), (idb, atb, db) = (load x, load y) in
  let format =
    match (da, db) with
    | Ssa_op.Decode.F32_exact, Ssa_op.Decode.F32_exact ->
        Contraction.Element.F32
    | Ssa_op.Decode.Bf16_to_f32, Ssa_op.Decode.Bf16_to_f32 ->
        Contraction.Element.Bf16
    | _ -> refuse Refusal.Format
  in
  if
    List.exists (Ssa_id.Buffer.equal ida) nest_stores
    || List.exists (Ssa_id.Buffer.equal idb) nest_stores
  then refuse Refusal.Operand_written;
  (* each axis: a variable, the sum's or a loop's (even one of a single
     value), or else zero *)
  let axes (at : Ssa_access.t) =
    match at with
    | Ssa_access.Flat _ -> refuse Refusal.Non_affine
    | Ssa_access.Coord c ->
        List.filter_map Fun.id
          (List.mapi
             (fun axis v ->
               let v = resolve v in
               if
                 key v = key kvar
                 || List.exists (fun l -> key l.iv = key v) loops
               then Some (axis, v)
               else
                 match known v with
                 | Some 0L -> None
                 | Some _ | None -> refuse Refusal.Non_affine)
             (Expr.Coord.to_list c))
  in
  let va = axes ata and vb = axes atb in
  let distinct l =
    let ks = List.map (fun (_, v) -> key v) l in
    List.length (List.sort_uniq compare ks) = List.length ks
  in
  if not (distinct va && distinct vb) then refuse Refusal.Non_affine;
  let axis_of l v =
    List.find_map (fun (a, w) -> if key w = key v then Some a else None) l
  in
  let mem l v = Option.is_some (axis_of l v) in
  let ka =
    match axis_of va kvar with Some a -> a | None -> refuse Refusal.Shape
  and kb =
    match axis_of vb kvar with Some a -> a | None -> refuse Refusal.Shape
  in
  let others l = List.filter (fun (_, v) -> key v <> key kvar) l in
  let batch = List.filter (fun (_, v) -> mem vb v) (others va) in
  let rows = List.filter (fun (_, v) -> not (mem vb v)) (others va) in
  let cols = List.filter (fun (_, v) -> not (mem va v)) (others vb) in
  if List.length batch > 1 then refuse Refusal.Batch_loops;
  if List.length rows > 2 || List.length cols > 1 then refuse Refusal.Shape;
  let one = function [ (_, v) ] -> Some v | _ -> None in
  let bvar = one batch and cvar = one cols in
  let extent = function
    | None -> 1L
    | Some v -> (
        match List.find_opt (fun l -> key l.iv = key v) loops with
        | Some { extent = Some e; _ } -> e
        | _ -> refuse Refusal.Bounds)
  in
  let stride_at id a =
    match buffer id with
    | Some (b : Ssa_buffer.t) -> List.nth (strides b.Ssa_buffer.extents) a
    | None -> refuse Refusal.Shape
  in
  let stride id l v =
    match v with
    | None -> 0L
    | Some v -> (
        match axis_of l v with
        | Some a -> stride_at id a
        | None -> refuse Refusal.Shape)
  in
  (* the rows, outer first: one variable, or two that flatten (the outer one's
     stride is the inner one's extent times its stride) *)
  let rows = List.sort (fun (a, _) (b, _) -> compare a b) rows in
  let row_vars = List.map snd rows in
  let m, a_outer =
    match rows with
    | [] -> (1L, 0L)
    | [ (a, v) ] -> (extent (Some v), stride_at ida a)
    | [ (a1, v1); (a2, v2) ] ->
        let e2 = extent (Some v2) in
        if Int64.equal (stride_at ida a1) (Int64.mul e2 (stride_at ida a2)) then
          (Int64.mul (extent (Some v1)) e2, stride_at ida a2)
        else refuse Refusal.Shape
    | _ -> refuse Refusal.Shape
  in
  let n = extent cvar and nb = extent bvar in
  let numerics = if !nested then Contraction.Numerics.Ordered else numerics in
  let desc =
    match
      Err.payload
        (Contraction.make ~batch:nb ~m ~n ~k
           ~a:
             {
               Contraction.Operand.batch = stride ida va bvar;
               outer = a_outer;
               k = stride_at ida ka;
             }
           ~b:
             {
               Contraction.Operand.batch = stride idb vb bvar;
               outer = stride idb vb cvar;
               k = stride_at idb kb;
             }
           ~c:{ Contraction.Output.batch = Int64.mul m n; row = n; col = 1L }
           ~format ~seed:Contraction.Seed.Zero ~numerics)
    with
    | Ok d -> d
    | Error _ -> refuse Refusal.Shape
  in
  let used = List.filter_map Fun.id [ bvar; cvar ] @ row_vars in
  let top =
    (* the outermost enclosing loop (the list is innermost first) *)
    List.fold_left
      (fun acc l ->
        if List.exists (fun v -> key v = key l.iv) used then Some l.iv else acc)
      None loops
  in
  let row_extents = List.map (fun v -> extent (Some v)) row_vars in
  {
    Found.desc;
    a = ida;
    b = idb;
    sum;
    batch = bvar;
    rows = List.combine row_vars row_extents;
    col = cvar;
    top;
  }

(* Every ordered sum of the program in program order, found or refused. *)
let census ~numerics (p : Ssa_program.t) =
  let consts = constants p in
  let ranges = Ssa_range.analyze p in
  let buffer = Ssa_program.find_buffer p in
  let out = ref [] in
  let rec region ~loops ~nest (r : Ssa_region.t) =
    List.iter (stmt ~loops ~nest) r.Ssa_region.body
  and stmt ~loops ~nest = function
    | Ssa_stmt.Instr _ -> ()
    | Ssa_stmt.If { then_; else_; _ } ->
        region ~loops ~nest then_;
        region ~loops ~nest else_
    | Ssa_stmt.For { lo; hi; step; body; _ } as s ->
        let iv = List.hd body.Ssa_region.params in
        let extent =
          match (index_const consts lo, index_const consts hi) with
          | Some 0L, Some h when step = 1L -> Some h
          | _ -> None
        in
        (* the stores of the outermost loop are the nest's *)
        let nest = match nest with None -> Some (stored [ s ]) | n -> n in
        region ~loops:({ iv; extent } :: loops) ~nest body
    | Ssa_stmt.Ordered_sum { lo; hi; seed; results; body; _ } as s -> (
        let nest_stores =
          match nest with Some st -> st | None -> stored [ s ]
        in
        match
          recognize ~numerics ~consts ~ranges ~buffer ~loops ~nest_stores
            (lo, hi, seed, results, body)
        with
        | f -> out := Outcome.Found f :: !out
        | exception Refuse r ->
            out := Outcome.Refused r :: !out;
            region ~loops ~nest body)
  in
  region ~loops:[] ~nest:None p.Ssa_program.entry;
  List.rev !out

(* {1 The rewrite}

   Each found product becomes one [Contract] into a fresh binary32 scratch
   buffer of extents [1, 1, 1, batch, m, n] ([1, 1, batch, h, w, n] when its
   rows are two loops, the same bytes), placed just before the outermost
   loop its variables belong to (before the sum when there is none) and
   threaded on that loop's effect; the sum becomes an in-bounds load of the
   scratch at its batch, row and column. Whatever the loops did with the sum
   (a bias, the store) stays. *)

let program ~numerics (p : Ssa_program.t) =
  let found =
    List.filter_map
      (function Outcome.Found f -> Some f | Outcome.Refused _ -> None)
      (census ~numerics p)
  in
  if found = [] then (p, [])
  else
    let next_buffer =
      1
      + List.fold_left
          (fun m (b : Ssa_buffer.t) -> max m (b.Ssa_buffer.id :> int))
          0 p.Ssa_program.buffers
    in
    let placed =
      List.mapi
        (fun i (f : Found.t) ->
          let d = f.Found.desc in
          let buffer =
            {
              Ssa_buffer.id = Ssa_id.Buffer.of_int (next_buffer + i);
              extents =
                Expr.Coord.of_fn (fun a ->
                    match (a, f.Found.rows) with
                    | Expr.Axis.C, _ -> d.Contraction.n
                    | Expr.Axis.D, [ _; _ ] | Expr.Axis.H, ([] | [ _ ]) ->
                        d.Contraction.batch
                    | Expr.Axis.H, [ (_, e); _ ] -> e
                    | Expr.Axis.W, ([] | [ _ ]) -> d.Contraction.m
                    | Expr.Axis.W, _ :: (_, e) :: _ -> e
                    | (Expr.Axis.D | Expr.Axis.N | Expr.Axis.T), _
                    | Expr.Axis.H, _ ->
                        1L);
              format = Ssa_format.F32;
              role = Ssa_buffer.Scratch;
            }
          in
          (f, buffer))
        found
    in
    let next = ref p.Ssa_program.next_value in
    let fresh ty =
      let id, n = Ssa_id.Value.Next.alloc !next in
      next := n;
      { Ssa_value.id; ty }
    in
    let instr ?token results op =
      Ssa_stmt.Instr
        { Ssa_instr.results; op; token; origin = Ssa_origin.Unknown }
    in
    (* the contractions before [token], and the effect after them *)
    let contracts fs token =
      List.fold_left
        (fun (stmts, token) ((f : Found.t), (s : Ssa_buffer.t)) ->
          let e = fresh Ssa_type.Effect in
          ( stmts
            @ [
                instr ~token [ e ]
                  (Ssa_op.Contract
                     {
                       desc = f.Found.desc;
                       a = f.Found.a;
                       b = f.Found.b;
                       c = s.Ssa_buffer.id;
                     });
              ],
            e ))
        ([], token) fs
    in
    let at_top iv =
      List.filter
        (fun ((f : Found.t), _) ->
          match f.Found.top with Some t -> key t = key iv | None -> false)
        placed
    in
    let of_sum (v : Ssa_value.t) =
      List.find_opt (fun ((f : Found.t), _) -> key f.Found.sum = key v) placed
    in
    let rec region (r : Ssa_region.t) =
      { r with Ssa_region.body = List.concat_map stmt r.Ssa_region.body }
    and stmt (s : Ssa_region.t Ssa_stmt.t) =
      match s with
      | Ssa_stmt.Instr _ -> [ s ]
      | Ssa_stmt.If i ->
          [
            Ssa_stmt.If
              { i with then_ = region i.then_; else_ = region i.else_ };
          ]
      | Ssa_stmt.For f -> (
          let body = region f.body in
          let iv = List.hd f.body.Ssa_region.params in
          match at_top iv with
          | [] -> [ Ssa_stmt.For { f with body } ]
          | fs ->
              let effect =
                List.find
                  (fun (v : Ssa_value.t) -> v.Ssa_value.ty = Ssa_type.Effect)
                  f.inits
              in
              let pre, token = contracts fs effect in
              let inits =
                List.map
                  (fun (v : Ssa_value.t) ->
                    if key v = key effect then token else v)
                  f.inits
              in
              pre @ [ Ssa_stmt.For { f with inits; body } ])
      | Ssa_stmt.Ordered_sum o -> (
          match o.results with
          | sum :: _ -> (
              match of_sum sum with
              | None -> [ Ssa_stmt.Ordered_sum { o with body = region o.body } ]
              | Some ((f : Found.t), scratch) ->
                  let pre, token =
                    match f.Found.top with
                    | None -> contracts [ (f, scratch) ] o.token
                    | Some _ -> ([], o.token)
                  in
                  let zero = fresh (Ssa_type.Scalar Ssa_type.Index) in
                  let coord v = match v with Some v -> v | None -> zero in
                  let row = List.map (fun (v, _) -> Some v) f.Found.rows in
                  let at =
                    Expr.Coord.of_fn (fun a ->
                        match (a, row) with
                        | Expr.Axis.C, _ -> coord f.Found.col
                        | Expr.Axis.D, [ _; _ ] | Expr.Axis.H, ([] | [ _ ]) ->
                            coord f.Found.batch
                        | Expr.Axis.H, [ r; _ ] -> coord r
                        | Expr.Axis.W, [ r ] | Expr.Axis.W, [ _; r ] -> coord r
                        | ( ( Expr.Axis.D | Expr.Axis.N | Expr.Axis.T
                            | Expr.Axis.H | Expr.Axis.W ),
                            _ ) ->
                            zero)
                  in
                  pre
                  @ [
                      instr [ zero ] (Ssa_op.Const (Ssa_const.Index 0L));
                      instr ~token o.results
                        (Ssa_op.Load_in_bounds
                           {
                             buffer = scratch.Ssa_buffer.id;
                             at = Ssa_access.Coord at;
                             decode = Ssa_op.Decode.F32_exact;
                           });
                    ])
          | [] -> [ s ])
    in
    let entry = region p.Ssa_program.entry in
    ( {
        p with
        Ssa_program.revision = Ssa_id.Revision.succ p.Ssa_program.revision;
        buffers = p.Ssa_program.buffers @ List.map snd placed;
        entry;
        next_value = !next;
      },
      found )
