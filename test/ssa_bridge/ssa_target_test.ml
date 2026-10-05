open Ssa_ir
open Loop_ir

(* The target descriptions and numerical policies are restated in the SSA
   library, which sees no Loop type. They must stay what the Loop planner's are:
   every target, at both precisions, on every operation. *)

let precisions =
  [
    (Ssa_numerics.Precision.F64, Loop_numerics.Precision.F64);
    (Ssa_numerics.Precision.F32, Loop_numerics.Precision.F32);
  ]

let targets =
  [
    (Ssa_target.neon128, Loop_target.neon128);
    (Ssa_target.scalar, Loop_target.scalar);
    (Ssa_target.wasm128, Loop_target.wasm128);
    (Ssa_target.wasm128_relaxed, Loop_target.wasm128_relaxed);
  ]

let support_name = function
  | Ssa_target.Native -> "native"
  | Ssa_target.Expanded -> "expanded"

let loop_support_name = function
  | Loop_target.Native -> "native"
  | Loop_target.Expanded -> "expanded"

let%expect_test "every target prices every operation as the Loop planner does" =
  let mismatches = ref [] in
  let note fmt = Fmt.kstr (fun s -> mismatches := s :: !mismatches) fmt in
  List.iter
    (fun (ssa0, loop0) ->
      List.iter
        (fun (sp, lp) ->
          let ssa = ssa0.Ssa_target.at sp and loop = loop0.Loop_target.at lp in
          let tag = Fmt.str "%s" loop.Loop_target.name in
          if ssa.Ssa_target.name <> loop.Loop_target.name then
            note "%s: name %s" tag ssa.Ssa_target.name;
          if ssa.Ssa_target.vector_bits <> loop.Loop_target.vector_bits then
            note "%s: vector_bits" tag;
          if
            Ssa_type.Lanes.to_int ssa.Ssa_target.lanes <> loop.Loop_target.lanes
          then note "%s: lanes" tag;
          if ssa.Ssa_target.inner_loops <> loop.Loop_target.inner_loops then
            note "%s: inner_loops" tag;
          if ssa.Ssa_target.fma <> loop.Loop_target.fma then note "%s: fma" tag;
          if ssa.Ssa_target.relaxed_madd <> loop.Loop_target.relaxed_madd then
            note "%s: relaxed_madd" tag;
          if ssa.Ssa_target.row_block <> loop.Loop_target.row_block then
            note "%s: row_block" tag;
          if
            Ssa_numerics.Precision.name ssa.Ssa_target.precision
            <> Loop_numerics.Precision.name loop.Loop_target.precision
          then note "%s: precision" tag;
          List.iter
            (fun lop ->
              let name = Loop_target.Op.name lop in
              match
                List.find_opt
                  (fun sop -> Ssa_target.Op.name sop = name)
                  Ssa_target.Op.all
              with
              | None -> note "%s: no operation %s" tag name
              | Some sop ->
                  if
                    support_name (ssa.Ssa_target.support sop)
                    <> loop_support_name (loop.Loop_target.support lop)
                  then note "%s: support of %s" tag name;
                  if
                    not
                      (Float.equal (ssa.Ssa_target.cost sop)
                         (loop.Loop_target.cost lop))
                  then note "%s: cost of %s" tag name)
            Loop_target.Op.all;
          if List.length Ssa_target.Op.all <> List.length Loop_target.Op.all
          then note "%s: operation lists differ" tag)
        precisions)
    targets;
  Fmt.pr "mismatches: %d@." (List.length !mismatches);
  List.iter (Fmt.pr "  %s@.") (List.rev !mismatches);
  [%expect {| mismatches: 0 |}]

let%expect_test "the numerical policies are the Loop planner's" =
  let rows =
    List.map2
      (fun s l ->
        ( Ssa_numerics.name s = Loop_numerics.name l,
          Ssa_numerics.identity s = Loop_numerics.identity l,
          Ssa_numerics.contraction_permitted s
          = Loop_numerics.contraction_permitted l,
          Ssa_numerics.reassociation_permitted s
          = Loop_numerics.reassociation_permitted l ))
      Ssa_numerics.all Loop_numerics.all
  in
  Fmt.pr "same policies, in order: %b@."
    (List.for_all (fun r -> r = (true, true, true, true)) rows);
  let backends =
    [
      (Ssa_numerics.Backend.C, Loop_numerics.Backend.C);
      (Ssa_numerics.Backend.Interpreter, Loop_numerics.Backend.Interpreter);
      (Ssa_numerics.Backend.Javascript, Loop_numerics.Backend.Javascript);
      (Ssa_numerics.Backend.Wasm, Loop_numerics.Backend.Wasm);
    ]
  in
  List.iter
    (fun (sb, lb) ->
      List.iter2
        (fun s l ->
          let a = Result.is_ok (Err.payload (Ssa_numerics.check ~backend:sb s))
          and b =
            Result.is_ok (Err.payload (Loop_numerics.check ~backend:lb l))
          in
          if a <> b then
            Fmt.pr "%s/%s differs@."
              (Ssa_numerics.Backend.name sb)
              (Ssa_numerics.name s))
        Ssa_numerics.all Loop_numerics.all)
    backends;
  (* the binary32 helpers agree bitwise on a spread of values *)
  let values =
    [ 0.; -0.; 0.1; -0.7; 1.; 3.14159; 1e-30; 1e30; nan; infinity ]
  in
  let same f g =
    List.for_all (fun x -> Core.Float_bits.equal_portable (f x) (g x)) values
  in
  Fmt.pr "erf32 agrees: %b@." (same Ssa_numerics.erf32 Loop_numerics.erf32);
  let triples =
    List.concat_map
      (fun a ->
        List.concat_map (fun b -> List.map (fun c -> (a, b, c)) values) values)
      values
  in
  Fmt.pr "fma32 agrees: %b@."
    (List.for_all
       (fun (a, b, c) ->
         let r = Ssa_const.round_f32 in
         let a = r a and b = r b and c = r c in
         Core.Float_bits.equal_portable (Ssa_numerics.fma32 a b c)
           (Loop_numerics.fma32 a b c))
       triples);
  [%expect
    {|
    same policies, in order: true
    erf32 agrees: true
    fma32 agrees: true |}]
