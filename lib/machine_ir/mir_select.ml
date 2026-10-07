(* The target-independent part of instruction selection: a builder that
   keeps every generic value's id and type, allocates fresh temporaries and
   instructions above the generic ones, threads order states through block
   splits, splits a block after a call that may fail, writes a failure record
   word by word from its kind's layout, and adds the record's runtime view to
   the selected program. A target supplies its own forms through the
   callbacks. *)

(* The largest value and instruction ids a generic function uses, plus one. *)
let watermarks (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) =
  let v = ref 0 and i = ref 0 in
  let see (x : Mir_value.t) =
    v := max !v (Mir_id.Value.to_int x.Mir_value.id)
  in
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      List.iter see b.Mir_block.params;
      see b.Mir_block.order;
      List.iter
        (fun (ins : Mir_op.t Mir_instr.t) ->
          i := max !i (Mir_id.Instr.to_int ins.Mir_instr.id);
          List.iter see ins.Mir_instr.results;
          Option.iter (fun o -> see o.Mir_order.output) ins.Mir_instr.order)
        b.Mir_block.body)
    f.Mir_func.blocks;
  (!v + 1, !i + 1)

(* Whether a call can fail: a helper that declares failures, or a function
   that can reach a [fail] or a fallible call (a cycle of calls is taken to
   fail). *)
let fallibility (p : Mir_program.generic) =
  let memo = Hashtbl.create 8 in
  let rec func id =
    match Hashtbl.find_opt memo (Mir_id.Func.to_int id) with
    | Some b -> b
    | None -> (
        Hashtbl.replace memo (Mir_id.Func.to_int id) true;
        match Mir_program.find_func p id with
        | None -> true
        | Some f ->
            let b =
              List.exists
                (fun (blk : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
                  (match blk.Mir_block.terminator with
                    | Mir_terminator.Fail _ -> true
                    | _ -> false)
                  || List.exists
                       (fun (i : Mir_op.t Mir_instr.t) ->
                         match i.Mir_instr.op with
                         | Mir_op.Call (c, _) -> callee c
                         | _ -> false)
                       blk.Mir_block.body)
                f.Mir_func.blocks
            in
            Hashtbl.replace memo (Mir_id.Func.to_int id) b;
            b)
  and callee = function
    | Mir_op.Callee.Func id -> func id
    | Mir_op.Callee.Helper id -> (
        match Mir_program.find_helper p id with
        | Some h -> h.Mir_helper.failures <> []
        | None -> true)
  in
  callee

type ('op, 'test) t = {
  defs : (int, Mir_op.t) Hashtbl.t;  (** generic value id -> defining op *)
  mutable next_value : int;
  mutable next_instr : int;
  mutable body : 'op Mir_sel.Op.t Mir_instr.t list;  (** reversed *)
  mutable origin : Mir_origin.t;
  orders : (int, Mir_value.t) Hashtbl.t;
      (** a generic order state replaced by a split block's order parameter *)
  mutable next_block : int;
  mutable cur_id : Mir_id.Block.t;
  mutable cur_params : Mir_value.t list;
  mutable cur_order : Mir_value.t;
  mutable finished :
    ('op Mir_sel.Op.t, 'test Mir_sel.Terminator.t) Mir_block.t list;
      (** reversed *)
  results : Mir_type.t list;  (** the generic function's own results *)
}

let create (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) =
  let next_value, next_instr = watermarks f in
  let defs = Hashtbl.create 256 in
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      List.iter
        (fun (ins : Mir_op.t Mir_instr.t) ->
          List.iter
            (fun (r : Mir_value.t) ->
              Hashtbl.replace defs
                (Mir_id.Value.to_int r.Mir_value.id)
                ins.Mir_instr.op)
            ins.Mir_instr.results)
        b.Mir_block.body)
    f.Mir_func.blocks;
  {
    defs;
    next_value;
    next_instr;
    body = [];
    origin = Mir_origin.unknown;
    orders = Hashtbl.create 8;
    next_block =
      1
      + List.fold_left
          (fun m (b : (_, _) Mir_block.t) ->
            max m (Mir_id.Block.to_int b.Mir_block.id))
          0 f.Mir_func.blocks;
    cur_id = f.Mir_func.entry;
    cur_params = [];
    cur_order = { Mir_value.id = Mir_id.Value.of_int 0; ty = Mir_type.Order };
    finished = [];
    results = f.Mir_func.results;
  }

(* The order state a generic one stands for after splits. *)
let ord b (v : Mir_value.t) =
  Option.value ~default:v
    (Hashtbl.find_opt b.orders (Mir_id.Value.to_int v.Mir_value.id))

let ord_edge b (e : Mir_edge.t) =
  { e with Mir_edge.order = ord b e.Mir_edge.order }

let fresh b ty =
  let id = Mir_id.Value.of_int b.next_value in
  b.next_value <- b.next_value + 1;
  { Mir_value.id; ty }

let push b ?order results op =
  let order =
    Option.map
      (fun o -> { o with Mir_order.input = ord b o.Mir_order.input })
      order
  in
  let id = Mir_id.Instr.of_int b.next_instr in
  b.next_instr <- b.next_instr + 1;
  b.body <- { Mir_instr.id; results; op; order; origin = b.origin } :: b.body

(* One machine instruction defining [result] (a fresh temporary of [ty] when
   no result is given). *)
let emit b ?result ty op =
  let r = match result with Some r -> r | None -> fresh b ty in
  push b [ r ] (Mir_sel.Op.Machine op);
  r

let const_of b (v : Mir_value.t) =
  match Hashtbl.find_opt b.defs (Mir_id.Value.to_int v.Mir_value.id) with
  | Some (Mir_op.Const c) -> Some c.Mir_const.bits
  | _ -> None

(* Closes the block under construction with [terminator]. *)
let finish b terminator =
  b.finished <-
    {
      Mir_block.id = b.cur_id;
      params = b.cur_params;
      order = b.cur_order;
      body = List.rev b.body;
      terminator;
    }
    :: b.finished

let start b id params order =
  b.cur_id <- id;
  b.cur_params <- params;
  b.cur_order <- order;
  b.body <- []

let fresh_block b =
  let id = Mir_id.Block.of_int b.next_block in
  b.next_block <- b.next_block + 1;
  id

(* After a call that may fail, whose order state is [after]: a branch on the
   status — [nonzero status] the target's test — to a block returning it, and
   construction continues in a fresh block whose order parameter stands for
   [after]. *)
let split_on_status b ~nonzero ~status ~after =
  let propagate = fresh_block b and continue = fresh_block b in
  let edge target = { Mir_edge.target; args = []; order = after } in
  finish b
    (Mir_sel.Terminator.Branch
       { test = nonzero status; then_ = edge propagate; else_ = edge continue });
  let o1 = fresh b Mir_type.Order in
  b.finished <-
    {
      Mir_block.id = propagate;
      params = [];
      order = o1;
      body = [];
      terminator =
        Mir_sel.Terminator.Return { Mir_return.values = [ status ]; order = o1 };
    }
    :: b.finished;
  let o2 = fresh b Mir_type.Order in
  Hashtbl.replace b.orders (Mir_id.Value.to_int after.Mir_value.id) o2;
  start b continue [] o2

(* The record of a failure, stored word by word through [store base offset
   value ~wide] (an ordered target store returning its op), then status 1.
   [const ty bits] materializes a constant; [float_bits] moves a float
   payload's bits to an integer register; [skip] leaves one word unstored (a
   mutation). [None] when a site-bearing kind finds no compatible entry. *)
let store_record b (f : Mir_fail.t) ~sites ~base ~const ~float_bits ~store ~one
    ?skip () =
  let failure = f.Mir_fail.failure in
  let site =
    if Mir_failure.uses_site failure then
      Mir_failure.bind_site ~table:sites failure |> Option.map Option.some
    else Some None
  in
  match site with
  | None -> None
  | Some site ->
      let order = ref (ord b f.Mir_fail.order) in
      let base = base () in
      let put ~wide off v =
        let output = fresh b Mir_type.Order in
        push b
          ~order:{ Mir_order.input = !order; output }
          []
          (Mir_sel.Op.Machine (store base off v ~wide));
        order := output
      in
      put ~wide:false Mir_failure.record_kind_offset
        (const Mir_type.i32 (Int64.of_int32 (Mir_failure.kind_word failure)));
      put ~wide:false Mir_failure.record_invocation_offset
        (const Mir_type.i32 (-1L));
      let layout = Mir_failure.layout failure in
      let payload = Array.of_list f.Mir_fail.payload in
      for k = 0 to Mir_failure.record_words - 1 do
        let word =
          match List.assoc_opt k layout with
          | None -> const Mir_type.i64 0L
          | Some (Mir_failure.Word.Const c) -> const Mir_type.i64 c
          | Some Mir_failure.Word.Site ->
              const Mir_type.i64
                (Int64.of_int (Mir_id.Site.to_int (Option.get site)))
          | Some (Mir_failure.Word.Payload p) -> (
              let v = payload.(p) in
              match v.Mir_value.ty with Mir_type.F64 -> float_bits v | _ -> v)
        in
        let skipped =
          match skip with
          | Some `Last -> (
              match List.rev layout with (j, _) :: _ -> j = k | [] -> k = 0)
          | None -> false
        in
        if not skipped then
          put ~wide:true (Mir_failure.record_word_offset k) word
      done;
      let status = one () in
      Some
        (Mir_sel.Terminator.Return
           { Mir_return.values = [ status ]; order = !order })

let record_region = Mir_id.Region.of_int 1_000_000
let record_view = Mir_id.View.of_int 1_000_000

(* The selected program: the generic one's objects plus the failure record's
   runtime view, and the selected functions. *)
let program (p : Mir_program.generic) funcs =
  {
    Mir_program.data_model = p.Mir_program.data_model;
    regions =
      p.Mir_program.regions
      @ [
          {
            Mir_region.id = record_region;
            size = Mir_failure.record_bytes;
            align = 16L;
            init = Mir_region.Uninitialized;
          };
        ];
    views =
      p.Mir_program.views
      @ [
          {
            Mir_view.id = record_view;
            region = record_region;
            offset = 0L;
            size = Mir_failure.record_bytes;
            perm = Mir_view.Read_write;
            role = Mir_view.Runtime;
            source = None;
          };
        ];
    helpers = p.Mir_program.helpers;
    funcs;
    main = p.Mir_program.main;
    planning = p.Mir_program.planning;
    revision = p.Mir_program.revision;
  }

(* The contraction mutation a target's evidence suite injects: a multiply
   whose result the next instruction adds becomes, there, one fused
   multiply-add. Over a generic body. *)
let contract (body : Mir_op.t Mir_instr.t list) =
  let rec go acc = function
    | ({
         Mir_instr.op = Mir_op.Fbinary (Mir_op.Fbinary.Mul, x, y);
         results = [ m ];
         _;
       } as mul)
      :: ({ Mir_instr.op = Mir_op.Fbinary (Mir_op.Fbinary.Add, a, b); _ } as add)
      :: rest
      when Mir_value.equal b m || Mir_value.equal a m ->
        let c = if Mir_value.equal b m then a else b in
        go
          ({ add with Mir_instr.op = Mir_op.Ffma (x, y, c) } :: mul :: acc)
          rest
    | i :: rest -> go (i :: acc) rest
    | [] -> List.rev acc
  in
  go [] body
