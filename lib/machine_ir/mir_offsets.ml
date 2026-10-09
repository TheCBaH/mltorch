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
  type t =
    | Dropped_term  (** an address's last loop-varying term left out *)
    | Doubled_bump  (** a loop-carried pointer advanced twice as far *)
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
    (* loop-carried pointers: per header, the new parameters, the arguments
       the loop's entry and its one back edge bind them to, and the bumps the
       back edge's block computes *)
    let iv_params = Hashtbl.create 4
    and iv_entry = Hashtbl.create 4
    and iv_back = Hashtbl.create 4
    and iv_bumps = Hashtbl.create 4
    and iv_made = Hashtbl.create 4 in
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
    (* The i32 header parameter [p] of [l] that its one back edge advances by
       a positive constant: the block of that edge, the step, the argument
       the entry edge binds it to, and the parameter's position. *)
    let induction (l : Mir_loop.t) (p : Mir_value.t) pre =
      match
        List.find_opt
          (fun (b : (_, _) Mir_block.t) ->
            bkey b.Mir_block.id = bkey l.Mir_loop.header)
          blocks
      with
      | None -> None
      | Some header -> (
          let rec index k = function
            | [] -> None
            | (x : Mir_value.t) :: rest ->
                if vkey x = vkey p then Some k else index (k + 1) rest
          in
          match index 0 header.Mir_block.params with
          | None -> None
          | Some k -> (
              let edges_into (b : (_, _) Mir_block.t) =
                List.filter
                  (fun (e : Mir_edge.t) ->
                    bkey e.Mir_edge.target = bkey l.Mir_loop.header)
                  (Mir_terminator.edges b.Mir_block.terminator)
              in
              let back =
                List.concat_map
                  (fun (b : (_, _) Mir_block.t) ->
                    if Mir_loop.mem l b.Mir_block.id then
                      List.map (fun e -> (b, e)) (edges_into b)
                    else [])
                  blocks
              and entry =
                List.concat_map
                  (fun (b : (_, _) Mir_block.t) ->
                    if bkey b.Mir_block.id = bkey pre then edges_into b else [])
                  blocks
              in
              match (back, entry) with
              | [ (latch, be) ], [ ee ] -> (
                  let const_of_step (x : Mir_value.t) =
                    match op_of x with
                    | Some (Mir_op.Const c) -> Some c.Mir_const.bits
                    | Some (Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W64, y))
                      -> (
                        match op_of y with
                        | Some (Mir_op.Const c)
                          when Mir_type.equal c.Mir_const.ty Mir_type.i32 ->
                            Some
                              (Mir_width.signed Mir_width.W32 c.Mir_const.bits)
                        | _ -> None)
                    | _ -> None
                  in
                  let is_p (x : Mir_value.t) =
                    vkey x = vkey p
                    ||
                    match op_of x with
                    | Some (Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W64, y)) ->
                        vkey y = vkey p
                    | _ -> false
                  in
                  (* the next value is [p + k] at 32 bits, or the i64 sum of
                     the extended counter and a constant, narrowed *)
                  let step =
                    match op_of (List.nth be.Mir_edge.args k) with
                    | Some (Mir_op.Iarith (Mir_op.Iarith.Add, a, b)) ->
                        if is_p a then const_of_step b else None
                    | Some (Mir_op.Narrow (Mir_width.W32, sum)) -> (
                        match op_of sum with
                        | Some (Mir_op.Iarith (Mir_op.Iarith.Add, a, b)) ->
                            if is_p a then const_of_step b else None
                        | _ -> None)
                    | _ -> None
                  in
                  match step with
                  | Some k0
                    when Int64.compare k0 0L > 0
                         && Int64.compare k0 0x10000L < 0 ->
                      Some (latch.Mir_block.id, k0, List.nth ee.Mir_edge.args k)
                  | _ -> None)
              | _ -> None))
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
    let shared = Hashtbl.create 16
    and shared_addr = Hashtbl.create 16
    and iteration = Hashtbl.create 16 in
    let subst = Hashtbl.create 16 in
    let term_key (l, c) =
      ( (match l with Leaf.Sext _ -> 0 | Leaf.Value _ -> 1),
        vkey (Leaf.source l),
        c )
    in
    let append_to p origin ty op =
      let pre =
        ref (Option.value ~default:[] (Hashtbl.find_opt appended (bkey p)))
      in
      let r = emit pre origin ty op in
      Hashtbl.replace appended (bkey p) !pre;
      r
    in
    let materialize_in p origin terms =
      let pre =
        ref (Option.value ~default:[] (Hashtbl.find_opt appended (bkey p)))
      in
      let r = materialize pre origin terms in
      Hashtbl.replace appended (bkey p) !pre;
      r
    in
    (* A pointer that is [hoisted + sext(iv) * c] on every iteration of [l],
       carried round its back edge: bound on the header, started in the
       preheader [pre] and advanced by the counter's step in the back edge's
       block. *)
    let induction_pointer (l : Mir_loop.t) pre origin hoisted iv c =
      match induction l iv pre with
      | None -> None
      | Some (latch, step, init) -> (
          let ikey = (bkey l.Mir_loop.header, vkey hoisted, c, vkey iv) in
          match Hashtbl.find_opt iv_made ikey with
          | Some ip -> Some ip
          | None ->
              let ip = fresh Mir_type.Ptr in
              let start =
                append_to pre origin Mir_type.Ptr
                  (Mir_op.Ptr_add
                     ( hoisted,
                       Option.get
                         (materialize_in pre origin [ (Leaf.Sext init, c) ]) ))
              in
              let bump = ref [] in
              let by =
                emit bump origin Mir_type.i64
                  (Mir_op.Const
                     (Mir_const.i64
                        (Int64.mul c
                           (if mutation = Some Mutation.Doubled_bump then
                              Int64.mul 2L step
                            else step))))
              in
              let next =
                emit bump origin Mir_type.Ptr (Mir_op.Ptr_add (ip, by))
              in
              let h = bkey l.Mir_loop.header in
              let add tbl v =
                Hashtbl.replace tbl h
                  (Option.value ~default:[] (Hashtbl.find_opt tbl h) @ [ v ])
              in
              add iv_params ip;
              add iv_entry start;
              add iv_back next;
              Hashtbl.replace iv_bumps (bkey latch)
                (Option.value ~default:[]
                   (Hashtbl.find_opt iv_bumps (bkey latch))
                @ List.rev !bump);
              Hashtbl.replace iv_made ikey ip;
              Some ip)
    in
    (* The loops around [b], outermost first, as far in as each has a
       preheader that lies in the loop outside it. *)
    let chain_of b =
      let outer_first = List.rev (Mir_loop.around loops b) in
      let rec go acc = function
        | [] -> List.rev acc
        | (l : Mir_loop.t) :: rest -> (
            match preheader l with
            | None -> []
            | Some p ->
                let ok =
                  match acc with
                  | [] -> true
                  | (outer, _) :: _ -> Mir_loop.mem outer p
                in
                if ok then go ((l, p) :: acc) rest else [])
      in
      go [] outer_first
    in
    List.iter
      (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
        match chain_of b.Mir_block.id with
        | [] -> ()
        | chain ->
            let depth = List.length chain in
            let level k = List.nth chain (k - 1) in
            (* how many loops of the chain hold the definition of [v] *)
            let home (v : Mir_value.t) =
              match Hashtbl.find_opt def_block (vkey v) with
              | None -> depth
              | Some d ->
                  List.length
                    (List.filter (fun (l, _) -> Mir_loop.mem l d) chain)
            in
            List.iter
              (fun (i : Mir_op.t Mir_instr.t) ->
                match i.Mir_instr.op with
                | Mir_op.Ptr_add (base, off) ->
                    let origin = i.Mir_instr.origin in
                    let base_view =
                      match op_of base with
                      | Some (Mir_op.Addr view) -> Some view
                      | _ -> None
                    in
                    let base_home =
                      match base_view with Some _ -> 0 | None -> home base
                    in
                    let fm = form off in
                    let eff (leaf, _) =
                      max (home (Leaf.source leaf)) base_home
                    in
                    let level_terms k =
                      List.filter (fun t -> eff t = k) fm.Form.terms
                    in
                    (* nothing to move unless some term lives outside the
                       innermost loop *)
                    if not (List.exists (fun t -> eff t < depth) fm.Form.terms)
                    then ()
                    else begin
                      let var =
                        match (mutation, List.rev (level_terms depth)) with
                        | Some Mutation.Dropped_term, _ :: rest -> List.rev rest
                        | _ -> level_terms depth
                      in
                      (* the pointer at level 0: the base, in the first
                         preheader when it is an address *)
                      let acc =
                        ref
                          (match base_view with
                          | Some view -> (
                              let key =
                                (bkey (snd (level 1)), Mir_id.View.to_int view)
                              in
                              match Hashtbl.find_opt shared_addr key with
                              | Some a -> a
                              | None ->
                                  let a =
                                    append_to
                                      (snd (level 1))
                                      origin Mir_type.Ptr (Mir_op.Addr view)
                                  in
                                  Hashtbl.replace shared_addr key a;
                                  a)
                          | None -> base)
                      in
                      for j = 0 to depth - 1 do
                        (* the terms whose leaves live at level [j] are added
                           where the next loop is entered; a counter of the
                           loop at level [j] is carried round it instead *)
                        let terms = level_terms j in
                        let terms =
                          if j = 0 then terms
                          else
                            let l, lpre = level j in
                            match
                              List.find_map
                                (fun ((leaf, c) as t) ->
                                  match leaf with
                                  | Leaf.Sext iv -> (
                                      match
                                        induction_pointer l lpre origin !acc iv
                                          c
                                      with
                                      | Some ip -> Some (t, ip)
                                      | None -> None)
                                  | Leaf.Value _ -> None)
                                terms
                            with
                            | Some (t, ip) ->
                                acc := ip;
                                List.filter (fun t' -> t' != t) terms
                            | None -> terms
                        in
                        if terms <> [] then begin
                          let pre = snd (level (j + 1)) in
                          let ikey_terms =
                            List.sort compare (List.map term_key terms)
                          in
                          let key = (bkey pre, vkey !acc, ikey_terms) in
                          acc :=
                            match Hashtbl.find_opt shared key with
                            | Some a -> a
                            | None ->
                                let sum =
                                  Option.get (materialize_in pre origin terms)
                                in
                                let a =
                                  append_to pre origin Mir_type.Ptr
                                    (Mir_op.Ptr_add (!acc, sum))
                                in
                                Hashtbl.replace shared key a;
                                a
                        end
                      done;
                      begin
                        let l, pre = level depth in
                        let hoisted = !acc in
                        let here = ref [] in
                        (* the pointer this iteration: shared by the block's
                           accesses with the same varying terms *)
                        let q =
                          match var with
                          | [] -> hoisted
                          | [ (Leaf.Sext iv, c) ]
                            when induction_pointer l pre origin hoisted iv c
                                 <> None ->
                              Option.get
                                (induction_pointer l pre origin hoisted iv c)
                          | _ -> (
                              let qkey =
                                ( bkey b.Mir_block.id,
                                  vkey hoisted,
                                  List.sort compare (List.map term_key var) )
                              in
                              match Hashtbl.find_opt iteration qkey with
                              | Some q -> q
                              | None ->
                                  (* the common factor last, as an element
                                     size the target's address form scales *)
                                  let g =
                                    List.fold_left
                                      (fun g (_, c) -> gcd g c)
                                      0L var
                                  in
                                  let sum =
                                    Option.get
                                      (materialize here origin
                                         (List.map
                                            (fun (l, c) -> (l, Int64.div c g))
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
                                               (Mir_op.Const (Mir_const.i64 g))
                                           ))
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
                                (Mir_op.Const (Mir_const.i64 fm.Form.constant))
                            in
                            List.rev
                              ({ i with Mir_instr.op = Mir_op.Ptr_add (q, k) }
                              :: !here)
                        in
                        Hashtbl.replace replaced
                          (Mir_id.Instr.to_int i.Mir_instr.id)
                          body
                      end
                    end
                | _ -> ())
              b.Mir_block.body)
      blocks;
    (* the loop-carried pointers: parameters on the header, arguments on its
       entry and back edges, and the bumps at the end of the back edge's block *)
    let with_pointers (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) =
      let params =
        b.Mir_block.params
        @ Option.value ~default:[]
            (Hashtbl.find_opt iv_params (bkey b.Mir_block.id))
      in
      let edge (e : Mir_edge.t) =
        match Hashtbl.find_opt iv_params (bkey e.Mir_edge.target) with
        | None -> e
        | Some _ ->
            let h = bkey e.Mir_edge.target in
            let extra =
              if
                List.exists
                  (fun (l : Mir_loop.t) ->
                    bkey l.Mir_loop.header = h && Mir_loop.mem l b.Mir_block.id)
                  loops
              then Hashtbl.find iv_back h
              else Hashtbl.find iv_entry h
            in
            { e with Mir_edge.args = e.Mir_edge.args @ extra }
      in
      let terminator =
        match b.Mir_block.terminator with
        | Mir_terminator.Branch br ->
            Mir_terminator.Branch
              {
                br with
                Mir_branch.then_ = edge br.Mir_branch.then_;
                else_ = edge br.Mir_branch.else_;
              }
        | Mir_terminator.Jump e -> Mir_terminator.Jump (edge e)
        | t -> t
      in
      { b with Mir_block.params; terminator }
    in
    let blocks =
      List.map
        (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
          let b = with_pointers b in
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
            @ Option.value ~default:[]
                (Hashtbl.find_opt iv_bumps (bkey b.Mir_block.id))
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
