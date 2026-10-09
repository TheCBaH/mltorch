(* Logical vectors split into the slices a register holds: every vector value
   of a generic program becomes [register_bytes / element bytes]-lane slices,
   logical lane [k] being lane [k mod per] of slice [k / per] — the lane map,
   the same for every value of one element type. Lanewise operations run per
   slice; a lane access per slice at its first lane's address; an extract or
   insert on the slice holding the lane. A precision conversion changes the
   lanes a slice holds, so it goes through halves: a widening converts each
   half of a narrow slice ([vslice]), a narrowing joins two converted halves
   ([vconcat]). The result is a generic program again, verified, so the
   generic interpreter compares it with the logical one. A mask is split like
   the vector it was compared from: its lane width is the compared elements',
   found from the value that makes it, so a mask whose elements cannot be found
   (a parameter no edge feeds a known mask) is refused, as are vectors that
   fill no whole number of registers and vectors crossing a call, return or
   failure payload. *)

module Refusal = struct
  type t =
    | Boundary of Mir_type.t
        (** a vector through a call, a return or a failure payload *)
    | Lanes of Mir_type.t  (** fills no whole number of registers *)
    | Mask of Mir_type.t
    | Operation of string  (** a vector operation the split does not take *)

  let pp fmt = function
    | Boundary ty ->
        Fmt.pf fmt "%a crosses a call, return or failure" Mir_type.pp ty
    | Lanes ty ->
        Fmt.pf fmt "%a fills no whole number of registers" Mir_type.pp ty
    | Mask ty -> Fmt.pf fmt "%a is not split" Mir_type.pp ty
    | Operation o -> Fmt.pf fmt "%s is not split" o
end

(* Fault injection for the evidence suite. No consumer passes one. *)
module Mutation = struct
  type t =
    | Stale_slice  (** a store's last slice written from its first *)
    | Swapped_slices  (** a load's first two slices exchanged *)
end

let is_vector (v : Mir_value.t) =
  match v.Mir_value.ty with
  | Mir_type.Vec _ | Mir_type.Mask _ -> true
  | _ -> false

let program ?mutation ~register_bytes (g : Mir_verify.Generic.t) =
  let mutated m = mutation = Some m in
  Err.Escape.with_escape @@ fun esc ->
  let refuse r = Err.Escape.throw esc r in
  let p = Mir_verify.Generic.program g in
  let per (e : Mir_type.Elem.t) =
    Int64.to_int (Int64.div register_bytes (Mir_type.Elem.bytes e))
  in
  (* a vector type's slice type and count *)
  let shape (ty : Mir_type.t) =
    match ty with
    | Mir_type.Vec (e, n) ->
        let k = per e and n = Mir_type.Lanes.to_int n in
        if k < 1 || n mod k <> 0 then refuse (Refusal.Lanes ty)
        else (Mir_type.Vec (e, Mir_type.Lanes.of_int k), n / k, k)
    | Mir_type.Mask _ -> refuse (Refusal.Mask ty)
    | _ -> invalid_arg "Mir_vsplit: a scalar has no slices"
  in
  let func (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) =
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
    (* every vector value's slices, made before any use is rewritten *)
    let slices = Hashtbl.create 64 in
    (* a mask's lane width, from the elements it was compared from *)
    let mask_elem : (int, Mir_type.Elem.t) Hashtbl.t = Hashtbl.create 16 in
    let key (v : Mir_value.t) = Mir_id.Value.to_int v.Mir_value.id in
    let vec_elem (v : Mir_value.t) =
      match v.Mir_value.ty with Mir_type.Vec (e, _) -> Some e | _ -> None
    in
    let elem_of (v : Mir_value.t) =
      match v.Mir_value.ty with
      | Mir_type.Mask _ -> Hashtbl.find_opt mask_elem (key v)
      | _ -> vec_elem v
    in
    let learn (r : Mir_value.t) e =
      match r.Mir_value.ty with
      | Mir_type.Mask _ when not (Hashtbl.mem mask_elem (key r)) ->
          Hashtbl.replace mask_elem (key r) e;
          true
      | _ -> false
    in
    (* to a fixpoint: a mask parameter learns from the edges that feed it *)
    let rec propagate () =
      let changed = ref false in
      let flow (target : Mir_id.Block.t) args =
        match
          List.find_opt
            (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
              Mir_id.Block.equal b.Mir_block.id target)
            f.Mir_func.blocks
        with
        | Some tb when List.length tb.Mir_block.params = List.length args ->
            List.iter2
              (fun (param : Mir_value.t) (arg : Mir_value.t) ->
                match elem_of arg with
                | Some e -> if learn param e then changed := true
                | None -> ())
              tb.Mir_block.params args
        | _ -> ()
      in
      List.iter
        (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
          List.iter
            (fun (i : Mir_op.t Mir_instr.t) ->
              match (i.Mir_instr.op, i.Mir_instr.results) with
              | Mir_op.Fcmp (_, a, _), [ r ] -> (
                  match vec_elem a with
                  | Some e -> if learn r e then changed := true
                  | None -> ())
              | (Mir_op.Pnot a | Mir_op.Copy a), [ r ] -> (
                  match elem_of a with
                  | Some e -> if learn r e then changed := true
                  | None -> ())
              | Mir_op.Pbinary (_, a, c), [ r ] -> (
                  (* the elements flow every way between the operands and the
                     result *)
                  match (elem_of a, elem_of c, elem_of r) with
                  | Some e, _, _ | None, Some e, _ | None, None, Some e ->
                      List.iter
                        (fun v -> if learn v e then changed := true)
                        [ a; c; r ]
                  | None, None, None -> ())
              | Mir_op.Select (m, a, _), [ r ] when is_vector m -> (
                  (* the arms' elements are the mask's *)
                  match (elem_of a, elem_of r) with
                  | Some e, _ | None, Some e ->
                      if learn m e then changed := true
                  | None, None -> ())
              | _ -> ())
            b.Mir_block.body;
          match b.Mir_block.terminator with
          | Mir_terminator.Branch br ->
              flow br.Mir_branch.then_.Mir_edge.target
                br.Mir_branch.then_.Mir_edge.args;
              flow br.Mir_branch.else_.Mir_edge.target
                br.Mir_branch.else_.Mir_edge.args
          | Mir_terminator.Jump e -> flow e.Mir_edge.target e.Mir_edge.args
          | Mir_terminator.Fail _ | Mir_terminator.Return _ -> ())
        f.Mir_func.blocks;
      if !changed then propagate ()
    in
    propagate ();
    (* a value's slice type and count: a mask's is its compared elements' *)
    let shape_of (v : Mir_value.t) =
      match v.Mir_value.ty with
      | Mir_type.Mask n -> (
          match Hashtbl.find_opt mask_elem (key v) with
          | None -> refuse (Refusal.Mask v.Mir_value.ty)
          | Some e ->
              let k = per e and n = Mir_type.Lanes.to_int n in
              if k < 1 || n mod k <> 0 then
                refuse (Refusal.Lanes v.Mir_value.ty)
              else (Mir_type.Mask (Mir_type.Lanes.of_int k), n / k, k))
      | ty -> shape ty
    in
    let note (v : Mir_value.t) =
      if is_vector v then
        let ty, count, _ = shape_of v in
        Hashtbl.replace slices
          (Mir_id.Value.to_int v.Mir_value.id)
          (Array.init count (fun _ -> fresh ty))
    in
    List.iter
      (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
        List.iter note b.Mir_block.params;
        List.iter
          (fun (i : Mir_op.t Mir_instr.t) -> List.iter note i.Mir_instr.results)
          b.Mir_block.body)
      f.Mir_func.blocks;
    let s (v : Mir_value.t) =
      Hashtbl.find slices (Mir_id.Value.to_int v.Mir_value.id)
    in
    let expand vs =
      List.concat_map
        (fun v -> if is_vector v then Array.to_list (s v) else [ v ])
        vs
    in
    let scalar_only vs =
      List.iter
        (fun (v : Mir_value.t) ->
          if is_vector v then refuse (Refusal.Boundary v.Mir_value.ty))
        vs
    in
    let edge (e : Mir_edge.t) =
      { e with Mir_edge.args = expand e.Mir_edge.args }
    in
    let block (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) =
      let out = ref [] in
      let emit ?(first = None) ?order (i : Mir_op.t Mir_instr.t) results op =
        out :=
          {
            Mir_instr.id = Option.value first ~default:(instr_id ());
            results;
            op;
            order;
            origin = i.Mir_instr.origin;
          }
          :: !out
      in
      (* [ty] lanes made by [op] *)
      let value (i : Mir_op.t Mir_instr.t) ty op =
        let r = fresh ty in
        emit i [ r ] op;
        r
      in
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          let vector =
            List.exists is_vector
              (Mir_op.operands i.Mir_instr.op @ i.Mir_instr.results)
          in
          if not vector then out := i :: !out
          else
            let result () =
              match i.Mir_instr.results with
              | [ r ] -> s r
              | _ -> invalid_arg "Mir_vsplit: a vector operation's result"
            in
            (* one operation per slice, at the slice's own lanes *)
            let per_slice f =
              Array.iteri (fun j r -> emit i [ r ] (f j)) (result ())
            in
            (* [count] accesses, chained on the order the original threads *)
            let chained count f =
              let o = Option.get i.Mir_instr.order in
              let input = ref o.Mir_order.input in
              for j = 0 to count - 1 do
                let output =
                  if j = count - 1 then o.Mir_order.output
                  else fresh Mir_type.Order
                in
                f j ~order:{ Mir_order.input = !input; output };
                input := output
              done
            in
            let lane_addr (acc : Mir_op.Vaccess.t) j k =
              if j = 0 then acc.Mir_op.Vaccess.addr
              else
                let off =
                  value i Mir_type.i64
                    (Mir_op.Const
                       (Mir_const.i64
                          (Int64.mul
                             (Int64.of_int (j * k))
                             acc.Mir_op.Vaccess.stride)))
                in
                value i Mir_type.Ptr
                  (Mir_op.Ptr_add (acc.Mir_op.Vaccess.addr, off))
            in
            match i.Mir_instr.op with
            | Mir_op.Copy a -> per_slice (fun j -> Mir_op.Copy (s a).(j))
            | Mir_op.Fbinary (o, a, b) ->
                per_slice (fun j -> Mir_op.Fbinary (o, (s a).(j), (s b).(j)))
            | Mir_op.Ffma (a, b, c) ->
                per_slice (fun j ->
                    Mir_op.Ffma ((s a).(j), (s b).(j), (s c).(j)))
            | Mir_op.Funary (o, a) ->
                per_slice (fun j -> Mir_op.Funary (o, (s a).(j)))
            | Mir_op.Fconvert (c, a) ->
                let src = s a and dst = result () in
                let sty, _, sk = shape a.Mir_value.ty in
                let dty, _, dk =
                  shape (List.hd i.Mir_instr.results).Mir_value.ty
                in
                let elem = function
                  | Mir_type.Vec (e, _) -> e
                  | _ -> invalid_arg "Mir_vsplit"
                in
                if sk = dk then
                  per_slice (fun j -> Mir_op.Fconvert (c, src.(j)))
                else if sk > dk then
                  (* widening: each half of a source slice *)
                  Array.iteri
                    (fun j r ->
                      let first = j * dk mod sk in
                      let half =
                        value i
                          (Mir_type.Vec (elem sty, Mir_type.Lanes.of_int dk))
                          (Mir_op.Vslice
                             ( Mir_type.Lane.of_int first,
                               Mir_type.Lanes.of_int dk,
                               src.(j * dk / sk) ))
                      in
                      emit i [ r ] (Mir_op.Fconvert (c, half)))
                    dst
                else
                  (* narrowing: the converted source slices joined *)
                  let parts = dk / sk in
                  Array.iteri
                    (fun j r ->
                      let converted =
                        List.init parts (fun q ->
                            value i
                              (Mir_type.Vec (elem dty, Mir_type.Lanes.of_int sk))
                              (Mir_op.Fconvert (c, src.((j * parts) + q))))
                      in
                      emit i [ r ] (Mir_op.Vconcat converted))
                    dst
            | Mir_op.Vextract (lane, a) ->
                let _, _, k = shape a.Mir_value.ty in
                let l = Mir_type.Lane.to_int lane in
                emit ~first:(Some i.Mir_instr.id) i i.Mir_instr.results
                  (Mir_op.Vextract
                     (Mir_type.Lane.of_int (l mod k), (s a).(l / k)))
            | Mir_op.Vinsert (lane, a, x) ->
                let _, _, k = shape a.Mir_value.ty in
                let l = Mir_type.Lane.to_int lane in
                per_slice (fun j ->
                    if j = l / k then
                      Mir_op.Vinsert
                        (Mir_type.Lane.of_int (l mod k), (s a).(j), x)
                    else Mir_op.Copy (s a).(j))
            | Mir_op.Vload acc ->
                let _, count, k =
                  shape (List.hd i.Mir_instr.results).Mir_value.ty
                in
                let rs = result () in
                chained count (fun j ~order ->
                    let at =
                      if mutated Mutation.Swapped_slices && count >= 2 && j < 2
                      then 1 - j
                      else j
                    in
                    let addr = lane_addr acc at k in
                    emit ~order i
                      [ rs.(j) ]
                      (Mir_op.Vload
                         {
                           acc with
                           Mir_op.Vaccess.lanes = Mir_type.Lanes.of_int k;
                           addr;
                         }))
            | Mir_op.Vsplat (_, x) ->
                let _, _, k = shape_of (List.hd i.Mir_instr.results) in
                per_slice (fun _ -> Mir_op.Vsplat (Mir_type.Lanes.of_int k, x))
            | Mir_op.Vstore (acc, v) ->
                let _, count, k = shape v.Mir_value.ty in
                chained count (fun j ~order ->
                    let addr = lane_addr acc j k in
                    emit ~order i []
                      (Mir_op.Vstore
                         ( {
                             acc with
                             Mir_op.Vaccess.lanes = Mir_type.Lanes.of_int k;
                             addr;
                           },
                           (s v).(if
                                    mutated Mutation.Stale_slice
                                    && j = count - 1
                                  then 0
                                  else j) )))
            | Mir_op.Call _ ->
                scalar_only
                  (Mir_op.operands i.Mir_instr.op @ i.Mir_instr.results)
            | Mir_op.Fcmp (c, a, b) ->
                per_slice (fun j -> Mir_op.Fcmp (c, (s a).(j), (s b).(j)))
            | Mir_op.Pbinary (o, a, b) ->
                per_slice (fun j -> Mir_op.Pbinary (o, (s a).(j), (s b).(j)))
            | Mir_op.Pnot a -> per_slice (fun j -> Mir_op.Pnot (s a).(j))
            | Mir_op.Select (m, a, b) ->
                (* a scalar predicate chooses whole vectors: it is every
                   slice's, a mask chooses lane by lane *)
                per_slice (fun j ->
                    Mir_op.Select
                      ( (if is_vector m then (s m).(j) else m),
                        (s a).(j),
                        (s b).(j) ))
            | op -> refuse (Refusal.Operation (Mir_op.name op)))
        b.Mir_block.body;
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
        | Mir_terminator.Fail fl as t ->
            scalar_only fl.Mir_fail.payload;
            t
        | Mir_terminator.Return r as t ->
            scalar_only r.Mir_return.values;
            t
      in
      {
        b with
        Mir_block.params = expand b.Mir_block.params;
        body = List.rev !out;
        terminator;
      }
    in
    { f with Mir_func.blocks = List.map block f.Mir_func.blocks }
  in
  let split =
    { p with Mir_program.funcs = List.map func p.Mir_program.funcs }
  in
  match Err.payload (Mir_verify.generic split) with
  | Ok g -> g
  | Error d -> invalid_arg (Fmt.str "Mir_vsplit: %a" Mir_diagnostic.pp d)
