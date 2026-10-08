(* Loop-invariant address terms computed once, before the loop. An address is
   a pointer plus an i64 byte offset; the offset, read through additions,
   subtractions, multiplications by a constant and sign extensions, is a sum
   of coefficients times leaves plus a constant. That reading is exact modulo
   2^64, and an extension of a [narrow] result is the narrowed value itself
   (out of range, the [narrow] is already a defect), so no bound is needed.
   In an innermost natural loop with a preheader (the header's one
   predecessor outside it, ending in a jump), the base and the terms whose
   leaves are defined outside the loop become one pointer at the end of the
   preheader, shared by the accesses with the same base and terms; inside, an
   address adds the varying terms (once per block for accesses that share
   them) and then its constant, which a target's address form takes as a
   displacement. Only the innermost loop: pointers moved further out would
   stay live through every loop inside them, one per access per level, on top
   of the counters. Pure instructions and narrowings nothing reads are then
   deleted. The result is verified again. *)

(* Fault injection for the evidence suite. No consumer passes one. *)
module Mutation = struct
  type t = Dropped_term  (** an address's last loop-varying term left out *)
end

module Leaf = struct
  type t =
    | Sext of Mir_value.t  (** an i32 value, sign-extended *)
    | Value of Mir_value.t  (** an i64 value *)

  let source = function Sext v | Value v -> v

  let compare a b =
    match (a, b) with
    | Sext x, Sext y | Value x, Value y -> Mir_value.compare x y
    | Sext _, Value _ -> -1
    | Value _, Sext _ -> 1
end

(* Coefficient-leaf terms (each leaf once, no zero coefficient) plus a
   constant, modulo 2^64. *)
module Form = struct
  type t = { terms : (Leaf.t * int64) list; constant : int64 }

  let const k = { terms = []; constant = k }
  let leaf l = { terms = [ (l, 1L) ]; constant = 0L }

  let add a b =
    let terms =
      List.fold_left
        (fun acc (l, c) ->
          match List.partition (fun (l', _) -> Leaf.compare l l' = 0) acc with
          | [ (_, c') ], rest ->
              let s = Int64.add c c' in
              if Int64.equal s 0L then rest else (l, s) :: rest
          | _ -> (l, c) :: acc)
        a.terms b.terms
    in
    { terms; constant = Int64.add a.constant b.constant }

  let scale k a =
    if Int64.equal k 0L then const 0L
    else
      {
        terms = List.map (fun (l, c) -> (l, Int64.mul k c)) a.terms;
        constant = Int64.mul k a.constant;
      }
end

let rec gcd a b =
  if Int64.equal b 0L then Int64.abs a else gcd b (Int64.rem a b)

(* pure instructions and narrowings nothing reads, to a fixpoint *)
let rec dead_code (blocks : (Mir_op.t, Mir_terminator.t) Mir_block.t list) =
  let vkey (v : Mir_value.t) = Mir_id.Value.to_int v.Mir_value.id in
  let used = Hashtbl.create 256 in
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          List.iter
            (fun v -> Hashtbl.replace used (vkey v) ())
            (Mir_op.operands i.Mir_instr.op))
        b.Mir_block.body;
      List.iter
        (fun v -> Hashtbl.replace used (vkey v) ())
        (Mir_terminator.operands b.Mir_block.terminator))
    blocks;
  let dead (i : Mir_op.t Mir_instr.t) =
    (match Mir_op.effect_class i.Mir_instr.op with
      | Mir_op.Effect.Pure -> true
      | Mir_op.Effect.Partial -> (
          match i.Mir_instr.op with Mir_op.Narrow _ -> true | _ -> false)
      | Mir_op.Effect.Call | Mir_op.Effect.Event | Mir_op.Effect.Read
      | Mir_op.Effect.Write ->
          false)
    && i.Mir_instr.order = None
    && List.for_all
         (fun v -> not (Hashtbl.mem used (vkey v)))
         i.Mir_instr.results
  in
  let removed = ref false in
  let blocks =
    List.map
      (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
        {
          b with
          Mir_block.body =
            List.filter
              (fun i ->
                if dead i then (
                  removed := true;
                  false)
                else true)
              b.Mir_block.body;
        })
      blocks
  in
  if !removed then dead_code blocks else blocks

let func ?mutation (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) =
  let next_value, next_instr = Mir_select.watermarks f in
  let next_value = ref next_value and next_instr = ref next_instr in
  let fresh ty =
    let id = Mir_id.Value.of_int !next_value in
    incr next_value;
    { Mir_value.id; ty }
  in
  let instr_id () =
    let id = Mir_id.Instr.of_int !next_instr in
    incr next_instr;
    id
  in
  let vkey (v : Mir_value.t) = Mir_id.Value.to_int v.Mir_value.id in
  let bkey = Mir_id.Block.to_int in
  (* one sweep: the rewritten blocks, and whether any address moved *)
  let sweep (blocks : (Mir_op.t, Mir_terminator.t) Mir_block.t list) =
    let loops =
      Mir_loop.find ~entry:f.Mir_func.entry
        (List.map
           (fun (b : (_, _) Mir_block.t) ->
             (b.Mir_block.id, Mir_terminator.successors b.Mir_block.terminator))
           blocks)
    in
    let def_block = Hashtbl.create 64 and def_op = Hashtbl.create 64 in
    List.iter
      (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
        List.iter
          (fun v -> Hashtbl.replace def_block (vkey v) b.Mir_block.id)
          b.Mir_block.params;
        List.iter
          (fun (i : Mir_op.t Mir_instr.t) ->
            List.iter
              (fun v ->
                Hashtbl.replace def_block (vkey v) b.Mir_block.id;
                Hashtbl.replace def_op (vkey v) i.Mir_instr.op)
              i.Mir_instr.results)
          b.Mir_block.body)
      blocks;
    let op_of v = Hashtbl.find_opt def_op (vkey v) in
    let i64_const v =
      match op_of v with
      | Some (Mir_op.Const c) when Mir_type.equal c.Mir_const.ty Mir_type.i64 ->
          Some c.Mir_const.bits
      | _ -> None
    in
    let rec form (v : Mir_value.t) =
      match op_of v with
      | Some (Mir_op.Const c) when Mir_type.equal c.Mir_const.ty Mir_type.i64 ->
          Form.const c.Mir_const.bits
      | Some (Mir_op.Iarith (Mir_op.Iarith.Add, a, b)) ->
          Form.add (form a) (form b)
      | Some (Mir_op.Iarith (Mir_op.Iarith.Sub, a, b)) ->
          Form.add (form a) (Form.scale (-1L) (form b))
      | Some (Mir_op.Iarith (Mir_op.Iarith.Mul, a, b)) -> (
          match (i64_const a, i64_const b) with
          | _, Some k -> Form.scale k (form a)
          | Some k, None -> Form.scale k (form b)
          | None, None -> Form.leaf (Leaf.Value v))
      | Some (Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W64, x)) -> (
          match op_of x with
          | Some (Mir_op.Narrow (_, y))
            when Mir_type.equal y.Mir_value.ty Mir_type.i64 ->
              form y
          | Some (Mir_op.Const { Mir_const.ty = Mir_type.Int w; bits }) ->
              Form.const (Mir_width.signed w bits)
          | _ -> Form.leaf (Leaf.Sext x))
      | _ -> Form.leaf (Leaf.Value v)
    in
    (* instructions appended to a preheader, and an address's replacement *)
    let appended = Hashtbl.create 4 and replaced = Hashtbl.create 16 in
    let preheader (l : Mir_loop.t) =
      match
        List.filter
          (fun (b : (_, _) Mir_block.t) ->
            (not (Mir_loop.mem l b.Mir_block.id))
            && List.exists
                 (fun s -> bkey s = bkey l.Mir_loop.header)
                 (Mir_terminator.successors b.Mir_block.terminator))
          blocks
      with
      | [ b ] -> (
          match b.Mir_block.terminator with
          | Mir_terminator.Jump _ -> Some b.Mir_block.id
          | _ -> None)
      | _ -> None
    in
    (* emits [op] into [out] (latest first), returning its result *)
    let emit out (origin : Mir_origin.t) ty op =
      let r = fresh ty in
      out :=
        {
          Mir_instr.id = instr_id ();
          results = [ r ];
          op;
          order = None;
          origin;
        }
        :: !out;
      r
    in
    let materialize out origin (terms : (Leaf.t * int64) list) =
      let term (l, c) =
        let x =
          match l with
          | Leaf.Sext v ->
              emit out origin Mir_type.i64
                (Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W64, v))
          | Leaf.Value v -> v
        in
        if Int64.equal c 1L then x
        else
          emit out origin Mir_type.i64
            (Mir_op.Iarith
               ( Mir_op.Iarith.Mul,
                 x,
                 emit out origin Mir_type.i64 (Mir_op.Const (Mir_const.i64 c))
               ))
      in
      match terms with
      | [] -> None
      | t :: rest ->
          Some
            (List.fold_left
               (fun acc t ->
                 emit out origin Mir_type.i64
                   (Mir_op.Iarith (Mir_op.Iarith.Add, acc, term t)))
               (term t) rest)
    in
    let shared = Hashtbl.create 16 and iteration = Hashtbl.create 16 in
    let subst = Hashtbl.create 16 in
    let term_key (l, c) =
      ( (match l with Leaf.Sext _ -> 0 | Leaf.Value _ -> 1),
        vkey (Leaf.source l),
        c )
    in
    List.iter
      (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
        match Mir_loop.around loops b.Mir_block.id with
        | [] -> ()
        | l :: _ -> (
            match preheader l with
            | None -> ()
            | Some p ->
                let outside v =
                  match Hashtbl.find_opt def_block (vkey v) with
                  | Some d -> not (Mir_loop.mem l d)
                  | None -> false
                in
                List.iter
                  (fun (i : Mir_op.t Mir_instr.t) ->
                    match i.Mir_instr.op with
                    | Mir_op.Ptr_add (base, off) -> (
                        let base_op =
                          match op_of base with
                          | Some (Mir_op.Addr view) -> Some (Mir_op.Addr view)
                          | _ when outside base -> None
                          | _ -> Some (Mir_op.Copy base)
                        in
                        let fm = form off in
                        let inv, var =
                          List.partition
                            (fun (leaf, _) -> outside (Leaf.source leaf))
                            fm.Form.terms
                        in
                        match base_op with
                        | Some (Mir_op.Copy _) -> ()
                        | _ when inv = [] -> ()
                        | _ ->
                            let origin = i.Mir_instr.origin in
                            (* one pointer per base and invariant terms in a
                               preheader; the constant stays a displacement *)
                            let hkey =
                              ( bkey p,
                                (match base_op with
                                | Some (Mir_op.Addr view) ->
                                    (0, Mir_id.View.to_int view)
                                | _ -> (1, vkey base)),
                                List.sort compare (List.map term_key inv) )
                            in
                            let hoisted =
                              match Hashtbl.find_opt shared hkey with
                              | Some h -> h
                              | None ->
                                  let pre =
                                    ref
                                      (Option.value ~default:[]
                                         (Hashtbl.find_opt appended (bkey p)))
                                  in
                                  let base' =
                                    match base_op with
                                    | Some op -> emit pre origin Mir_type.Ptr op
                                    | None -> base
                                  in
                                  let h =
                                    emit pre origin Mir_type.Ptr
                                      (Mir_op.Ptr_add
                                         ( base',
                                           Option.get
                                             (materialize pre origin inv) ))
                                  in
                                  Hashtbl.replace appended (bkey p) !pre;
                                  Hashtbl.replace shared hkey h;
                                  h
                            in
                            let var =
                              match (mutation, List.rev var) with
                              | Some Mutation.Dropped_term, _ :: rest ->
                                  List.rev rest
                              | _ -> var
                            in
                            let here = ref [] in
                            (* the pointer this iteration: shared by the
                               block's accesses with the same varying terms *)
                            let q =
                              match var with
                              | [] -> hoisted
                              | _ -> (
                                  let qkey =
                                    ( bkey b.Mir_block.id,
                                      vkey hoisted,
                                      List.sort compare (List.map term_key var)
                                    )
                                  in
                                  match Hashtbl.find_opt iteration qkey with
                                  | Some q -> q
                                  | None ->
                                      (* the common factor last, as an element
                                         size the target's address form
                                         scales *)
                                      let g =
                                        List.fold_left
                                          (fun g (_, c) -> gcd g c)
                                          0L var
                                      in
                                      let sum =
                                        Option.get
                                          (materialize here origin
                                             (List.map
                                                (fun (l, c) ->
                                                  (l, Int64.div c g))
                                                var))
                                      in
                                      let bytes =
                                        if Int64.equal g 1L then sum
                                        else
                                          emit here origin Mir_type.i64
                                            (Mir_op.Iarith
                                               ( Mir_op.Iarith.Mul,
                                                 sum,
                                                 emit here origin Mir_type.i64
                                                   (Mir_op.Const
                                                      (Mir_const.i64 g)) ))
                                      in
                                      let q =
                                        emit here origin Mir_type.Ptr
                                          (Mir_op.Ptr_add (hoisted, bytes))
                                      in
                                      Hashtbl.replace iteration qkey q;
                                      q)
                            in
                            let body =
                              if Int64.equal fm.Form.constant 0L then (
                                List.iter
                                  (fun r -> Hashtbl.replace subst (vkey r) q)
                                  i.Mir_instr.results;
                                List.rev !here)
                              else
                                let k =
                                  emit here origin Mir_type.i64
                                    (Mir_op.Const
                                       (Mir_const.i64 fm.Form.constant))
                                in
                                List.rev
                                  ({
                                     i with
                                     Mir_instr.op = Mir_op.Ptr_add (q, k);
                                   }
                                  :: !here)
                            in
                            Hashtbl.replace replaced
                              (Mir_id.Instr.to_int i.Mir_instr.id)
                              body)
                    | _ -> ())
                  b.Mir_block.body))
      blocks;
    let blocks =
      List.map
        (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
          let body =
            List.concat_map
              (fun (i : Mir_op.t Mir_instr.t) ->
                match
                  Hashtbl.find_opt replaced (Mir_id.Instr.to_int i.Mir_instr.id)
                with
                | Some is -> is
                | None -> [ i ])
              b.Mir_block.body
            @ List.rev
                (Option.value ~default:[]
                   (Hashtbl.find_opt appended (bkey b.Mir_block.id)))
          in
          let sub v =
            Option.value ~default:v (Hashtbl.find_opt subst (vkey v))
          in
          let edge (e : Mir_edge.t) =
            { e with Mir_edge.args = List.map sub e.Mir_edge.args }
          in
          {
            b with
            Mir_block.body =
              List.map
                (fun (i : Mir_op.t Mir_instr.t) ->
                  {
                    i with
                    Mir_instr.op = Mir_op.map_operands sub i.Mir_instr.op;
                  })
                body;
            terminator =
              (match b.Mir_block.terminator with
              | Mir_terminator.Branch br ->
                  Mir_terminator.Branch
                    {
                      Mir_branch.cond = sub br.Mir_branch.cond;
                      then_ = edge br.Mir_branch.then_;
                      else_ = edge br.Mir_branch.else_;
                    }
              | Mir_terminator.Fail f ->
                  Mir_terminator.Fail
                    {
                      f with
                      Mir_fail.payload = List.map sub f.Mir_fail.payload;
                    }
              | Mir_terminator.Jump e -> Mir_terminator.Jump (edge e)
              | Mir_terminator.Return r ->
                  Mir_terminator.Return
                    {
                      r with
                      Mir_return.values = List.map sub r.Mir_return.values;
                    });
          })
        blocks
    in
    blocks
  in
  let blocks = sweep f.Mir_func.blocks in
  { f with Mir_func.blocks = dead_code blocks }

let program ?mutation (g : Mir_verify.Generic.t) =
  let p = Mir_verify.Generic.program g in
  Err.or_raise ~pp_error:Mir_diagnostic.pp
    (Mir_verify.generic
       {
         p with
         Mir_program.funcs = List.map (func ?mutation) p.Mir_program.funcs;
       })
