open Inference_compare_schema
module Plan = Inference_compare_plan
module Ex = Inference_compare_extract
module E = Inference_compare_engines
module P = Plan_fixtures
module C = Chain_fixture

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let local (r : Tensor_ref.t) = Id.Tensor_local.to_string r.local_id
let nlocal (r : Node_ref.t) = Id.Node_local.to_string r.local_id

let describe (e : Ex.Extraction.t) =
  Format.printf "%s/%s: inputs=[%s] outputs=[%s] nodes=[%s]@."
    (Id.Unit.to_string e.unit) (Side.to_string e.side)
    (String.concat "," (List.map local e.inputs))
    (String.concat "," (List.map local e.outputs))
    (String.concat ","
       (List.map
          (fun (n : Node_desc.t) ->
            Printf.sprintf "%s(%s)->(%s)" (nlocal n.node)
              (String.concat "," (List.map local n.operands))
              (String.concat "," (List.map local n.results)))
          e.body.nodes))

let extract_all s =
  match Ex.Extraction.extract_all ~hashers ~algorithm s with
  | Ok l -> l
  | Error e ->
      Format.kasprintf failwith "extract: %a" Ex.Extract_error.pp
        (Err.Error.kind e)

let find l u side =
  List.find
    (fun (e : Ex.Extraction.t) -> Id.Unit.to_string e.unit = u && e.side = side)
    l

let%expect_test "units become ordinary self-contained graphs" =
  let s = C.structural C.attention in
  let all = extract_all s in
  List.iter describe
    (List.filter (fun (e : Ex.Extraction.t) -> e.side = Left) all);
  [%expect
    {|
    u_mm/left: inputs=[x,w] outputs=[m] nodes=[n_mm(x,w)->(m)]
    u_add/left: inputs=[m,x] outputs=[a] nodes=[n_add(m,x)->(a)]
    u_mul/left: inputs=[m] outputs=[mm] nodes=[n_mul(m,m)->(mm)]
    u_sm/left: inputs=[a] outputs=[s] nodes=[n_sm(a)->(s)]
    u_att/left: inputs=[s,w] outputs=[o] nodes=[n_att(s,w)->(o)]
    u_tail/left: inputs=[o] outputs=[t] nodes=[n_rs(o)->(r),n_sum(r)->(t)] |}]

let%expect_test "the id maps and digests" =
  let all = extract_all (C.structural C.attention) in
  let e = find all "u_tail" Left in
  let d = e.descriptor in
  Format.printf "extracted version: %a@." Id.Graph_version.pp
    d.extracted_key.version;
  Format.printf "source graph kept: %b, digest differs from the source: %b@."
    (Graph_key.equal d.source_graph C.left_key)
    (not
       (Digest.Graph.equal d.extracted_key.graph_digest C.left_key.graph_digest));
  Format.printf "tensor map: %a@."
    Fmt.(
      list ~sep:comma
        (using
           (fun ((a : Tensor_ref.t), (b : Tensor_ref.t)) ->
             local a ^ "->" ^ local b ^ "@"
             ^ String.concat "/"
                 (List.map Name.Scope_component.to_string b.scope))
           string))
    d.original_to_extracted_tensors;
  Format.printf "node map: %a@."
    Fmt.(
      list ~sep:comma
        (using
           (fun ((a : Node_ref.t), (b : Node_ref.t)) ->
             nlocal a ^ "->" ^ nlocal b)
           string))
    d.original_to_extracted_nodes;
  (* left and right of one unit are different graphs; repeated extraction is
     identical *)
  let again = find (extract_all (C.structural C.attention)) "u_tail" Left in
  Format.printf "repeat extraction identical payload digest: %b@."
    (Digest.Extracted_graph_payload.equal d.graph_payload_digest
       again.descriptor.graph_payload_digest);
  Format.printf "left and right payloads differ: %b@."
    (not
       (Digest.Extracted_graph_payload.equal d.graph_payload_digest
          (find all "u_tail" Right).descriptor.graph_payload_digest));
  [%expect
    {|
    extracted version: v1#u_tail:left
    source graph kept: true, digest differs from the source: true
    tensor map: o->o@u_tail, r->r@u_tail,
    t->t@u_tail
    node map: n_rs->n_rs,
    n_sum->n_sum
    repeat extraction identical payload digest: true
    left and right payloads differ: true |}]

let%expect_test "repeated operands, multi-output nodes and passthroughs" =
  (* mul(m, m) keeps both operand occurrences. *)
  let all = extract_all (C.structural C.attention) in
  describe (find all "u_mul" Left);
  (* dup has two results; the passthrough unit returns one of its inputs. *)
  let t =
    C.synthetic
      [
        C.node "n_dup" "dup" [ "x" ] [ "a"; "b" ];
        C.node "n_add" "add" [ "a"; "b" ] [ "c" ];
      ]
      ~sources:[ "x" ] ~finals:[ "c" ]
      ~units:[ ("U", [ "n_dup"; "n_add" ]) ]
  in
  describe (find (extract_all (C.structural t)) "U" Left);
  (* a unit that forwards its input as a declared output *)
  let pass =
    C.synthetic
      [ C.node "n1" "id" [ "x" ] [ "h" ]; C.node "n2" "neg" [ "h" ] [ "y" ] ]
      ~sources:[ "x" ] ~finals:[ "y"; "h" ]
      ~units:[ ("A", [ "n1" ]); ("B", [ "n2" ]) ]
  in
  describe (find (extract_all (C.structural pass)) "B" Left);
  [%expect
    {|
    u_mul/left: inputs=[m] outputs=[mm] nodes=[n_mul(m,m)->(mm)]
    U/left: inputs=[x] outputs=[c] nodes=[n_dup(x)->(a,b),n_add(a,b)->(c)]
    B/left: inputs=[h] outputs=[y] nodes=[n2(h)->(y)] |}]

let%expect_test "noncontiguous clusters and zero-arithmetic assertions" =
  (* U = {n1, n3} with n2 (unit V) listed between them. *)
  let t =
    C.synthetic
      [
        C.node "n1" "id" [ "x" ] [ "p" ];
        C.node "n2" "neg" [ "w" ] [ "q" ];
        C.node "n3" "add" [ "p"; "x" ] [ "r" ];
      ]
      ~sources:[ "x"; "w" ] ~finals:[ "r"; "q" ]
      ~units:[ ("U", [ "n1"; "n3" ]); ("V", [ "n2" ]) ]
  in
  describe (find (extract_all (C.structural t)) "U" Left);
  (* an assertion unit: no results, no outputs *)
  let a =
    C.synthetic
      [
        C.node "n1" "id" [ "x" ] [ "y" ];
        C.node "chk" "assert_nonneg" [ "x" ] [] ~assertion:true;
      ]
      ~sources:[ "x" ] ~finals:[ "y" ]
      ~units:[ ("U", [ "n1" ]); ("CHK", [ "chk" ]) ]
  in
  let all = extract_all (C.structural a) in
  describe (find all "CHK" Left);
  let chk = find all "CHK" Left in
  Format.printf "assertion effect preserved: %b@."
    (List.for_all
       (fun (n : Node_desc.t) -> n.side_effect = Metadata_assertion)
       chk.body.nodes);
  [%expect
    {|
    U/left: inputs=[x] outputs=[r] nodes=[n1(x)->(p),n3(p,x)->(r)]
    CHK/left: inputs=[x] outputs=[] nodes=[chk(x)->()]
    assertion effect preserved: true |}]

let%expect_test "dialect realization reconciles the inventory" =
  let s = C.structural C.u1u2 in
  let e = find (extract_all s) "U1" Left in
  let realize fault =
    let module D =
      (val E.Synthetic_dialect.make ?fault ()
          : Ex.Dialect.S with type graph = Graph_desc.t
           and type body = Graph_desc.t)
    in
    match
      Ex.Dialect.realize
        (module D)
        (Plan.Graph_index.graph (Plan.Structural.left s))
        e
    with
    | Ok _ -> "accepted"
    | Error err ->
        Format.asprintf "refused: %s"
          (match (Err.Error.kind err).code with
          | Output_inventory_failed -> "output_inventory_failed"
          | Boundary_mismatch -> "boundary_mismatch"
          | _ -> "other")
  in
  Format.printf "faithful dialect: %s@." (realize None);
  Format.printf "drops an output: %s@." (realize (Some Drop_last_output));
  Format.printf "renames an operator: %s@." (realize (Some Rename_operator));
  (* an internal tensor exposed as an extra output *)
  let chain =
    C.synthetic
      [ C.node "n1" "id" [ "x" ] [ "p" ]; C.node "n2" "neg" [ "p" ] [ "q" ] ]
      ~sources:[ "x" ] ~finals:[ "q" ]
      ~units:[ ("U", [ "n1"; "n2" ]) ]
  in
  let sc = C.structural chain in
  let ec = find (extract_all sc) "U" Left in
  let module Dx =
    (val E.Synthetic_dialect.make ~fault:Expose_extra_output ()
        : Ex.Dialect.S with type graph = Graph_desc.t
         and type body = Graph_desc.t)
  in
  (match
     Ex.Dialect.realize
       (module Dx)
       (Plan.Graph_index.graph (Plan.Structural.left sc))
       ec
   with
  | Ok _ -> Format.printf "exposes an undeclared tensor: accepted@."
  | Error err ->
      Format.printf "exposes an undeclared tensor: refused: %s@."
        (match (Err.Error.kind err).code with
        | Output_inventory_failed -> "output_inventory_failed"
        | _ -> "other"));
  (* two outputs reordered *)
  let two =
    C.synthetic
      [ C.node "n1" "dup" [ "x" ] [ "a"; "b" ] ]
      ~sources:[ "x" ] ~finals:[ "a"; "b" ]
      ~units:[ ("U", [ "n1" ]) ]
  in
  let s2 = C.structural two in
  let e2 = find (extract_all s2) "U" Left in
  let module D2 =
    (val E.Synthetic_dialect.make ~fault:Reorder_outputs ()
        : Ex.Dialect.S with type graph = Graph_desc.t
         and type body = Graph_desc.t)
  in
  (match
     Ex.Dialect.realize
       (module D2)
       (Plan.Graph_index.graph (Plan.Structural.left s2))
       e2
   with
  | Ok _ -> Format.printf "reorders outputs: accepted@."
  | Error err ->
      Format.printf "reorders outputs: refused: %s@."
        (match (Err.Error.kind err).code with
        | Output_inventory_failed -> "output_inventory_failed"
        | _ -> "other"));
  [%expect
    {|
    faithful dialect: accepted
    drops an output: refused: output_inventory_failed
    renames an operator: refused: boundary_mismatch
    exposes an undeclared tensor: refused: output_inventory_failed
    reorders outputs: refused: output_inventory_failed |}]

let%expect_test "a declared passthrough points at an input" =
  let s = C.structural C.u1u2 in
  let b =
    Option.get
      (Plan.Partition_check.boundary
         (Plan.Structural.partition s)
         (P.uid "U1") Left)
  in
  let x = List.hd b.inputs in
  let b' = { b with outputs = b.outputs @ [ x ] } in
  (match Ex.Extraction.plan ~hashers ~algorithm (Plan.Structural.left s) b' with
  | Ok e ->
      describe e;
      Format.printf "outputs include the input itself: %b@."
        (List.exists (fun o -> Tensor_ref.equal o (List.hd e.inputs)) e.outputs)
  | Error err ->
      Format.printf "ERROR %a@." Ex.Extract_error.pp (Err.Error.kind err));
  [%expect
    {|
    U1/left: inputs=[x] outputs=[h,x] nodes=[n1(x)->(h)]
    outputs include the input itself: true |}]
