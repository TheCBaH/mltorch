open Machine_ir
open Machine_interp
open Machine_target_aarch64
open A64_op
module Sel = A64_stage.Sel

(* M5: hand-built selected forms interpreted on any host, and malformed form,
   class, immediate, condition-state and feature requests rejected. *)

let v n ty = { Mir_value.id = Mir_id.Value.of_int n; ty }
let o n = v n Mir_type.Order
let i64 = Mir_type.i64
let i32 = Mir_type.i32

(* One block: [params], then [body] as (results, op), returning [results]. *)
let program ?(features = [ Mir_target.Feature.Fp ]) ?(extra = []) params body
    results =
  let instrs =
    List.mapi
      (fun k (rs, op) ->
        {
          Mir_instr.id = Mir_id.Instr.of_int k;
          results = rs;
          op = Mir_sel.Op.Machine op;
          order = None;
          origin = Mir_origin.unknown;
        })
      body
  in
  let entry =
    {
      Mir_block.id = Mir_id.Block.of_int 0;
      params;
      order = o 999;
      body = instrs;
      terminator =
        Mir_sel.Terminator.Return { Mir_return.values = results; order = o 999 };
    }
  in
  {
    Sel.features;
    program =
      {
        Mir_program.data_model = Mir_layout.Data_model.Lp64_le;
        regions = [];
        views = [];
        helpers = [];
        funcs =
          [
            {
              Mir_func.id = Mir_id.Func.of_int 0;
              name = "t";
              entry = Mir_id.Block.of_int 0;
              results =
                List.map (fun (r : Mir_value.t) -> r.Mir_value.ty) results;
              blocks = entry :: extra;
            };
          ];
        main = Mir_id.Func.of_int 0;
        planning = None;
        revision = Mir_id.Revision.of_int 0;
      };
  }

let run sel args =
  match Err.payload (Sel.verify sel) with
  | Error d -> Fmt.pr "rejected: %a@." Mir_diagnostic.pp d
  | Ok vsel ->
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate sel.Sel.program memory ~bound:(fun _ -> None))
      in
      let r = A64_stage.Interp.run vsel memory binding ~args in
      Fmt.pr "%a@." Mir_interp.Outcome.pp r.A64_stage.Interp.outcome

let%expect_test "hand-built forms execute" =
  (* 0x1234_5678_9abc_def0 by movz/movk, a W add that wraps and zero-extends,
     an unsigned and a signed compare of the same bits *)
  let a = v 0 i64 and w = v 1 i32 in
  run
    (program [ a; w ]
       [
         ([ v 10 i64 ], Movz (i64, 0xdef0, 0));
         ([ v 11 i64 ], Movk (Sz.X, v 10 i64, 0x9abc, 16));
         ([ v 12 i64 ], Movk (Sz.X, v 11 i64, 0x5678, 32));
         ([ v 13 i64 ], Movk (Sz.X, v 12 i64, 0x1234, 48));
         ([ v 14 i32 ], Add (Sz.W, w, w));
         ([ v 15 Mir_type.Flags ], Cmp (Sz.W, v 14 i32, w));
         ([ v 16 Mir_type.Pred ], Cset (Cond.Lo, v 15 Mir_type.Flags));
         ([ v 17 Mir_type.Flags ], Cmp (Sz.W, v 14 i32, w));
         ([ v 18 Mir_type.Pred ], Cset (Cond.Lt, v 17 Mir_type.Flags));
       ]
       [ v 13 i64; v 14 i32; v 16 Mir_type.Pred; v 18 Mir_type.Pred ])
    [ Mir_datum.Bits 0L; Mir_datum.Bits 0x8000_0001L ];
  (* fcmp of NaN: unordered sets C and V; mi (ordered less) is false, lt true *)
  let x = v 0 Mir_type.F64 and y = v 1 Mir_type.F64 in
  run
    (program [ x; y ]
       [
         ([ v 10 Mir_type.Flags ], Fcmp (Fsz.D, x, y));
         ([ v 11 Mir_type.Pred ], Cset (Cond.Mi, v 10 Mir_type.Flags));
         ([ v 12 Mir_type.Flags ], Fcmp (Fsz.D, x, y));
         ([ v 13 Mir_type.Pred ], Cset (Cond.Lt, v 12 Mir_type.Flags));
         ([ v 14 Mir_type.F64 ], Fmadd (Fsz.D, x, y, y));
         ([ v 15 i64 ], Fcvtzs (Fsz.D, x));
       ]
       [ v 11 Mir_type.Pred; v 13 Mir_type.Pred; v 14 Mir_type.F64; v 15 i64 ])
    [ Mir_datum.f64 Float.nan; Mir_datum.f64 1. ];
  (* sdiv by zero and min / -1 are defined *)
  let p = v 0 i64 and q = v 1 i64 in
  List.iter
    (fun (a, b) ->
      run
        (program [ p; q ] [ ([ v 10 i64 ], Sdiv (Sz.X, p, q)) ] [ v 10 i64 ])
        [ Mir_datum.Bits a; Mir_datum.Bits b ])
    [ (7L, 0L); (Int64.min_int, -1L); (-7L, 2L) ];
  [%expect
    {|
    success [0x123456789abcdef0, 0x2, 0x1, 0x0]
    success [0x0, 0x1, 0x7ff8000000000001, 0x0]
    success [0x0]
    success [0x8000000000000000]
    success [0xfffffffffffffffd] |}]

let%expect_test "malformed selected programs are rejected" =
  let a = v 0 i64 and w = v 1 i32 and x = v 2 Mir_type.F64 in
  let show label sel =
    Fmt.pr "%s: " label;
    run sel []
  in
  show "class"
    (program [ a; w; x ] [ ([ v 10 i64 ], Add (Sz.X, w, w)) ] [ v 10 i64 ]);
  show "immediate"
    (program [ a; w; x ]
       [ ([ v 10 i64 ], Add_imm (Sz.X, a, 5000L)) ]
       [ v 10 i64 ]);
  show "bitmask"
    (program [ a; w; x ]
       [ ([ v 10 i32 ], Logic_imm (Logic.And, Sz.W, w, 5L)) ]
       [ v 10 i32 ]);
  show "result type"
    (program [ a; w; x ] [ ([ v 10 i32 ], Add (Sz.X, a, a)) ] [ v 10 i32 ]);
  show "condition across a clobber"
    (program [ a; w; x ]
       [
         ([ v 10 Mir_type.Flags ], Cmp (Sz.X, a, a));
         ([ v 11 Mir_type.Flags ], Cmp (Sz.W, w, w));
         ([ v 12 Mir_type.Pred ], Cset (Cond.Eq, v 10 Mir_type.Flags));
       ]
       [ v 12 Mir_type.Pred ]);
  show "missing feature"
    (program ~features:[] [ a; w; x ]
       [ ([ v 10 Mir_type.F64 ], Fbin (Fop.Add, Fsz.D, x, x)) ]
       [ v 10 Mir_type.F64 ]);
  show "tied operand types"
    (program [ a; w; x ] [ ([ v 10 i64 ], Movk (Sz.W, a, 1, 0)) ] [ v 10 i64 ]);
  (* the condition defined in the entry and tested by the next block's branch *)
  let next =
    {
      Mir_block.id = Mir_id.Block.of_int 1;
      params = [];
      order = o 998;
      body = [];
      terminator =
        Mir_sel.Terminator.Return { Mir_return.values = []; order = o 998 };
    }
  in
  let sel =
    program ~extra:[ next ] [ a; w; x ]
      [ ([ v 10 Mir_type.Flags ], Cmp (Sz.X, a, a)) ]
      []
  in
  let f = List.hd sel.Sel.program.Mir_program.funcs in
  let entry = List.hd f.Mir_func.blocks in
  let edge =
    { Mir_edge.target = Mir_id.Block.of_int 1; args = []; order = o 999 }
  in
  let entry =
    { entry with Mir_block.terminator = Mir_sel.Terminator.Jump edge }
  in
  let branch =
    {
      next with
      Mir_block.terminator =
        Mir_sel.Terminator.Branch
          {
            test = B_cond (Cond.Eq, v 10 Mir_type.Flags);
            then_ =
              {
                edge with
                Mir_edge.target = Mir_id.Block.of_int 2;
                order = o 998;
              };
            else_ =
              {
                edge with
                Mir_edge.target = Mir_id.Block.of_int 2;
                order = o 998;
              };
          };
    }
  in
  let exit_ =
    {
      next with
      Mir_block.id = Mir_id.Block.of_int 2;
      order = o 997;
      terminator =
        Mir_sel.Terminator.Return { Mir_return.values = []; order = o 997 };
    }
  in
  show "condition across a block"
    {
      sel with
      Sel.program =
        {
          sel.Sel.program with
          Mir_program.funcs =
            [
              {
                f with
                Mir_func.results = [];
                blocks = [ entry; branch; exit_ ];
              };
            ];
        };
    };
  show "branch on a non-condition"
    {
      sel with
      Sel.program =
        {
          sel.Sel.program with
          Mir_program.funcs =
            [
              {
                f with
                Mir_func.results = [];
                blocks =
                  [
                    entry;
                    {
                      branch with
                      Mir_block.terminator =
                        Mir_sel.Terminator.Branch
                          {
                            test = B_cond (Cond.Eq, w);
                            then_ =
                              {
                                edge with
                                Mir_edge.target = Mir_id.Block.of_int 2;
                                order = o 998;
                              };
                            else_ =
                              {
                                edge with
                                Mir_edge.target = Mir_id.Block.of_int 2;
                                order = o 998;
                              };
                          };
                    };
                    exit_;
                  ];
              };
            ];
        };
    };
  [%expect
    {|
    class: rejected: selected fn0 bb0 i0: target constraint: add operands
    immediate: rejected: selected fn0 bb0 i0: target constraint: add immediate
    bitmask: rejected: selected fn0 bb0 i0: target constraint: bitmask immediate
    result type: rejected: selected fn0 bb0 i0: results do not match the opcode
    condition across a clobber: rejected: selected fn0 bb0 i2: target constraint: condition state used across a block or a clobber
    missing feature: rejected: selected fn0 bb0 i0: target constraint: needs feature fp
    tied operand types: rejected: selected fn0 bb0 i0: target constraint: movk operand
    condition across a block: rejected: selected fn0 bb1: target constraint: condition state used across a block or a clobber
    branch on a non-condition: rejected: selected fn0 bb1: target constraint: branch on a non-condition |}]

let%expect_test "an unsigned compare selected as signed is detected" =
  let bld = Mir_builder.create () in
  let e = Mir_builder.new_block bld [ i64; i64 ] in
  let x, y =
    match Mir_builder.param e with [ x; y ] -> (x, y) | _ -> assert false
  in
  let lt = Mir_builder.emit bld e (Mir_op.Icmp (Mir_op.Icmp.Ult, x, y)) in
  Mir_builder.return e [ lt ];
  let p =
    Mir_builder.program
      [
        Mir_builder.func bld ~id:(Mir_id.Func.of_int 0) ~name:"ult" ~entry:e
          ~results:[ Mir_type.Pred ];
      ]
      ~main:(Mir_id.Func.of_int 0)
  in
  let args = [ Mir_datum.Bits (-1L); Mir_datum.Bits 0L ] in
  print_endline (A64_harness.generic_program p ~args);
  print_endline
    (A64_harness.generic_program ~mutation:A64_select.Mutation.Signed_compare p
       ~args);
  [%expect {|
    [0x0]
    DISAGREE generic [0x0] vs aarch64 [0x1] |}]
