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
    row : Ssa_value.t option;
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

let recognize ~numerics ~consts ~buffer ~(loops : loop list) ~nest_stores
    (lo, hi, seed, results, (body : Ssa_region.t)) =
  let sum =
    match results with [ s; _ ] -> s | _ -> refuse Refusal.Not_product
  in
  (match sum.Ssa_value.ty with
  | Ssa_type.Scalar Ssa_type.F32 -> ()
  | _ -> refuse Refusal.Binary64);
  (match index_const consts lo with
  | Some 0L -> ()
  | _ -> refuse Refusal.Bounds);
  let k =
    match index_const consts hi with
    | Some k -> k
    | None -> refuse Refusal.Bounds
  in
  (match Hashtbl.find_opt consts (key seed) with
  | Some (Ssa_const.F32 x) when x = 0. && 1. /. x > 0. -> ()
  | _ -> refuse Refusal.Seed);
  let kvar =
    match body.Ssa_region.params with
    | iv :: _ -> iv
    | [] -> refuse Refusal.Shape
  in
  let term =
    match body.Ssa_region.yields with
    | [ t; _ ] -> t
    | _ -> refuse Refusal.Not_product
  in
  (* the body: marks, two binary32 loads and their product *)
  let loads = Hashtbl.create 4 and product = ref None in
  List.iter
    (function
      | Ssa_stmt.Instr { Ssa_instr.op = Ssa_op.Mark Ssa_mark.Reduction; _ } ->
          ()
      | Ssa_stmt.Instr
          {
            Ssa_instr.op =
              Ssa_op.Load_in_bounds
                { buffer = id; at; decode = Ssa_op.Decode.F32_exact };
            results = [ v; _ ];
            _;
          } ->
          Hashtbl.replace loads (key v) (id, at)
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
      | _ -> refuse Refusal.Not_product)
    body.Ssa_region.body;
  let x, y =
    match !product with Some p -> p | None -> refuse Refusal.Not_product
  in
  let load v =
    match Hashtbl.find_opt loads (key v) with
    | Some l -> l
    | None -> refuse Refusal.Not_product
  in
  let (ida, ata), (idb, atb) = (load x, load y) in
  if
    List.exists (Ssa_id.Buffer.equal ida) nest_stores
    || List.exists (Ssa_id.Buffer.equal idb) nest_stores
  then refuse Refusal.Operand_written;
  (* each axis: a constant zero, or a variable: the sum's or a loop's *)
  let axes (at : Ssa_access.t) =
    match at with
    | Ssa_access.Flat _ -> refuse Refusal.Non_affine
    | Ssa_access.Coord c ->
        List.filter_map Fun.id
          (List.mapi
             (fun axis v ->
               match index_const consts v with
               | Some 0L -> None
               | Some _ -> refuse Refusal.Non_affine
               | None ->
                   if
                     key v = key kvar
                     || List.exists (fun l -> key l.iv = key v) loops
                   then Some (axis, v)
                   else refuse Refusal.Non_affine)
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
  if List.length rows > 1 || List.length cols > 1 then refuse Refusal.Shape;
  let one = function [ (_, v) ] -> Some v | _ -> None in
  let bvar = one batch and rvar = one rows and cvar = one cols in
  let extent = function
    | None -> 1L
    | Some v -> (
        match List.find_opt (fun l -> key l.iv = key v) loops with
        | Some { extent = Some e; _ } -> e
        | _ -> refuse Refusal.Bounds)
  in
  let stride id l v =
    match v with
    | None -> 0L
    | Some v -> (
        match (axis_of l v, buffer id) with
        | Some a, Some (b : Ssa_buffer.t) ->
            List.nth (strides b.Ssa_buffer.extents) a
        | _ -> refuse Refusal.Shape)
  in
  let kstride id a =
    match buffer id with
    | Some (b : Ssa_buffer.t) -> List.nth (strides b.Ssa_buffer.extents) a
    | None -> refuse Refusal.Shape
  in
  let m = extent rvar and n = extent cvar and nb = extent bvar in
  let desc =
    match
      Err.payload
        (Contraction.make ~batch:nb ~m ~n ~k
           ~a:
             {
               Contraction.Operand.batch = stride ida va bvar;
               outer = stride ida va rvar;
               k = kstride ida ka;
             }
           ~b:
             {
               Contraction.Operand.batch = stride idb vb bvar;
               outer = stride idb vb cvar;
               k = kstride idb kb;
             }
           ~c:{ Contraction.Output.batch = Int64.mul m n; row = n; col = 1L }
           ~format:Contraction.Element.F32 ~seed:Contraction.Seed.Zero ~numerics)
    with
    | Ok d -> d
    | Error _ -> refuse Refusal.Shape
  in
  let used = List.filter_map Fun.id [ bvar; rvar; cvar ] in
  let top =
    (* the outermost enclosing loop (the list is innermost first) *)
    List.fold_left
      (fun acc l ->
        if List.exists (fun v -> key v = key l.iv) used then Some l.iv else acc)
      None loops
  in
  {
    Found.desc;
    a = ida;
    b = idb;
    sum;
    batch = bvar;
    row = rvar;
    col = cvar;
    top;
  }

(* Every ordered sum of the program in program order, found or refused. *)
let census ~numerics (p : Ssa_program.t) =
  let consts = constants p in
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
    | Ssa_stmt.Ordered_sum { lo; hi; seed; results; body; _ } as s ->
        let nest_stores =
          match nest with Some st -> st | None -> stored [ s ]
        in
        (out :=
           match
             recognize ~numerics ~consts ~buffer ~loops ~nest_stores
               (lo, hi, seed, results, body)
           with
           | f -> Outcome.Found f :: !out
           | exception Refuse r -> Outcome.Refused r :: !out);
        region ~loops ~nest body
  in
  region ~loops:[] ~nest:None p.Ssa_program.entry;
  List.rev !out

(* {1 The rewrite}

   Each found product becomes one [Contract] into a fresh binary32 scratch
   buffer of extents [1, 1, 1, batch, m, n], placed just before the outermost
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
                    match a with
                    | Expr.Axis.H -> d.Contraction.batch
                    | Expr.Axis.W -> d.Contraction.m
                    | Expr.Axis.C -> d.Contraction.n
                    | Expr.Axis.N | Expr.Axis.T | Expr.Axis.D -> 1L);
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
                  let at =
                    Expr.Coord.of_fn (fun a ->
                        match a with
                        | Expr.Axis.H -> coord f.Found.batch
                        | Expr.Axis.W -> coord f.Found.row
                        | Expr.Axis.C -> coord f.Found.col
                        | Expr.Axis.N | Expr.Axis.T | Expr.Axis.D -> zero)
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
