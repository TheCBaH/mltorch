open Machine_ir
open Mir_fixtures

(* The generic verifier: the fixtures verify, and each way a program can be
   malformed is rejected with a diagnostic naming the stage and location. *)

let verdict p =
  match Err.payload (Mir_verify.generic p) with
  | Ok _ -> "ok"
  | Error d -> Fmt.str "%a" Mir_diagnostic.pp d

let%expect_test "hand-built fixtures verify" =
  List.iter
    (fun p -> print_endline (verdict p))
    [ swap (); sum_f32 (); div (); store_f32 () ];
  [%expect {|
    ok
    ok
    ok
    ok |}]

(* Rewrite every block of the main function. *)
let map_blocks f (p : Mir_program.generic) =
  {
    p with
    Mir_program.funcs =
      List.map
        (fun (fn : (_, _) Mir_func.t) ->
          { fn with Mir_func.blocks = List.map f fn.Mir_func.blocks })
        p.Mir_program.funcs;
  }

let map_instrs f =
  map_blocks (fun (b : (_, _) Mir_block.t) ->
      { b with Mir_block.body = List.map f b.Mir_block.body })

(* The first instruction matching [pick] rewritten by [f]. *)
let map_first pick f p =
  let hit = ref false in
  map_instrs
    (fun (i : Mir_op.t Mir_instr.t) ->
      if (not !hit) && pick i.Mir_instr.op then (
        hit := true;
        f i)
      else i)
    p

let is_op name (op : Mir_op.t) = Mir_op.name op = name

let%expect_test "malformed programs are rejected" =
  let show label p = Fmt.pr "%s: %s@." label (verdict p) in
  (* a second definition of an existing value *)
  show "duplicate definition"
    (map_first (is_op "add")
       (fun i ->
         let first = List.hd i.Mir_instr.results in
         ignore first;
         {
           i with
           Mir_instr.results =
             [
               {
                 (List.hd i.Mir_instr.results) with
                 Mir_value.id = Mir_id.Value.of_int 0;
               };
             ];
         })
       (swap ()));
  (* an i64 operand where the comparison's other side is an i32 *)
  show "wrong width"
    (map_first (is_op "icmp.slt")
       (fun i ->
         match i.Mir_instr.op with
         | Mir_op.Icmp (c, a, b) ->
             {
               i with
               Mir_instr.op =
                 Mir_op.Icmp (c, a, { b with Mir_value.ty = Mir_type.i64 });
             }
         | _ -> i)
       (swap ()));
  (* integer arithmetic on a pointer *)
  show "pointer arithmetic"
    (map_first (is_op "ptr.add")
       (fun i ->
         match i.Mir_instr.op with
         | Mir_op.Ptr_add (base, off) ->
             {
               i with
               Mir_instr.op = Mir_op.Iarith (Mir_op.Iarith.Add, base, off);
             }
         | _ -> i)
       (sum_f32 ()));
  (* an edge that drops an argument *)
  show "missing edge argument"
    (map_blocks
       (fun (b : (_, _) Mir_block.t) ->
         match b.Mir_block.terminator with
         | Mir_terminator.Jump e when List.length e.Mir_edge.args = 3 ->
             {
               b with
               Mir_block.terminator =
                 Mir_terminator.Jump
                   { e with Mir_edge.args = List.tl e.Mir_edge.args };
             }
         | _ -> b)
       (swap ()));
  (* the event consumes the order state the store already consumed *)
  show "stale order"
    (map_first (is_op "event")
       (fun i ->
         match i.Mir_instr.order with
         | Some o ->
             {
               i with
               Mir_instr.order =
                 Some
                   {
                     o with
                     Mir_order.input =
                       {
                         o.Mir_order.input with
                         Mir_value.id = Mir_id.Value.of_int 2;
                       };
                   };
             }
         | None -> i)
       (store_f32 ()));
  (* a helper whose signature takes an order state *)
  show "bad helper signature"
    {
      (div ()) with
      Mir_program.helpers =
        [
          {
            Mir_helper.id = Mir_id.Helper.of_int 0;
            name = "exp";
            version = 1;
            params = [ Mir_type.Order ];
            results = [ Mir_type.F64 ];
            effects = Mir_helper.Effect.Pure;
            failures = [];
          };
        ];
    };
  (* a coordinate failure with five coordinates *)
  show "malformed payload"
    (map_blocks
       (fun (b : (_, _) Mir_block.t) ->
         match b.Mir_block.terminator with
         | Mir_terminator.Fail f ->
             {
               b with
               Mir_block.terminator =
                 Mir_terminator.Fail
                   { f with Mir_fail.payload = List.tl f.Mir_fail.payload };
             }
         | _ -> b)
       (sum_f32 ()));
  (* a division with no zero guard dominating it *)
  show "unguarded division"
    (map_blocks
       (fun (b : (_, _) Mir_block.t) ->
         match b.Mir_block.terminator with
         | Mir_terminator.Branch br ->
             (* swap the zero branch's arms: the division now runs on the zero
                side *)
             {
               b with
               Mir_block.terminator =
                 Mir_terminator.Branch
                   {
                     br with
                     Mir_branch.then_ = br.Mir_branch.else_;
                     else_ = br.Mir_branch.then_;
                   };
             }
         | _ -> b)
       (div ()));
  [%expect
    {|
    duplicate definition: generic fn0 bb2 i5: %0 is defined twice
    wrong width: generic fn0 bb1 i3: %0 has a type its role forbids
    pointer arithmetic: generic fn0 bb3 i11: operand 0 has type ptr64
    missing edge argument: generic fn0 bb0: edge to bb1 passes 2 arguments for 3 parameters
    stale order: generic fn0 bb0 i3: stale order state %2
    bad helper signature: generic: malformed helper helper0
    malformed payload: generic fn0 bb4: failure coord_out_of_range(t7, W): payload does not match its schema
    unguarded division: generic fn0 bb3 i7: a domain-restricted operation without guard evidence |}]

let%expect_test "mutated use lists and effects cannot fool the verifier" =
  let show label p = Fmt.pr "%s: %s@." label (verdict p) in
  (* a store recorded without its order state: the opcode is still ordered *)
  show "store without order"
    (map_first (is_op "store.i32")
       (fun i -> { i with Mir_instr.order = None })
       (store_f32 ()));
  (* a pure operation given order state *)
  show "pure with order"
    (map_first (is_op "bitcast")
       (fun i ->
         let o =
           { Mir_value.id = Mir_id.Value.of_int 1; ty = Mir_type.Order }
         in
         {
           i with
           Mir_instr.order =
             Some
               {
                 Mir_order.input = o;
                 output = { o with Mir_value.id = Mir_id.Value.of_int 999 };
               };
         })
       (store_f32 ()));
  (* an operand renamed to a value nothing defines *)
  show "undefined operand"
    (map_first (is_op "fadd")
       (fun i ->
         {
           i with
           Mir_instr.op =
             Mir_op.map_operands
               (fun v -> { v with Mir_value.id = Mir_id.Value.of_int 4242 })
               i.Mir_instr.op;
         })
       (sum_f32 ()));
  (* a result whose recorded type disagrees with the opcode *)
  show "result type"
    (map_first (is_op "bitcast")
       (fun i ->
         {
           i with
           Mir_instr.results =
             List.map
               (fun r -> { r with Mir_value.ty = Mir_type.F64 })
               i.Mir_instr.results;
         })
       (sum_f32 ()));
  (* a store through a read-only view *)
  let ro =
    let bld = Mir_builder.create () in
    let e = Mir_builder.new_block bld [] in
    let a = Mir_builder.emit bld e (Mir_op.Addr (Mir_id.View.of_int 0)) in
    let z = k32 bld e 0L in
    Mir_builder.emit_unit bld e
      (Mir_op.Store
         ({ Mir_op.Access.width = Mir_width.W32; addr = a; align = 4L }, z));
    Mir_builder.return e [];
    program
      ~regions:
        [
          {
            Mir_region.id = Mir_id.Region.of_int 0;
            size = 4L;
            align = 4L;
            init = Mir_region.Bound;
          };
        ]
      ~views:
        [
          {
            Mir_view.id = Mir_id.View.of_int 0;
            region = Mir_id.Region.of_int 0;
            offset = 0L;
            size = 4L;
            perm = Mir_view.Read;
            role = Mir_view.Input;
            source = None;
          };
        ]
      (Mir_builder.func bld ~id:fn ~name:"ro" ~entry:e ~results:[])
  in
  show "read-only store" ro;
  [%expect
    {|
    store without order: generic fn0 bb0 i2: an ordered operation without order state
    pure with order: generic fn0 bb0 i1: an unordered operation with order state
    undefined operand: generic fn0 bb3 i15: %4242 is never defined
    result type: generic fn0 bb3 i13: results do not match the opcode
    read-only store: generic fn0 bb0 i2: an access view0 does not permit |}]
