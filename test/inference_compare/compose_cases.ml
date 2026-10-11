open Inference_compare_schema
module F = Compose_fixture
module P = Plan_fixtures
module Plan = Inference_compare_plan
module Compose_error = Inference_compare_compose.Compose_error

let pp_plan = P.pp_error
let ok_or what = function Ok v -> v | Error _ -> failwith what
let names l = List.map Id.Unit.to_string l

(* ---- three graphs of the same value: renamed, reshaped, fused, split ------ *)

let n s g = s ^ "_" ^ g

(* X: h = x + w; y = h * 100. *)
let gx =
  F.graph F.key_x
    [
      ("n1", "add", [ "x"; "w" ], [ "h" ]); ("n2", "mulc:100", [ "h" ], [ "y" ]);
    ]
    ~sources:[ "x"; "w" ] ~finals:[ "y" ]

let renamed ?shape key g =
  F.graph ?shape key
    [
      (n "n1" g, "add", [ n "x" g; n "w" g ], [ n "h" g ]);
      (n "n2" g, "mulc:100", [ n "h" g ], [ n "y" g ]);
    ]
    ~sources:[ n "x" g; n "w" g ]
    ~finals:[ n "y" g ]

let gy = renamed ~shape:[ 3L; 1L ] F.key_y "y"
let gz = renamed F.key_z "z"
let gw = renamed ~shape:[ 3L; 1L ] F.key_w "w"

(* X's names have no suffix. *)
let map_xy =
  {
    F.left = gx;
    right = gy;
    units = [ ("u1", [ "n1" ], [ "n1_y" ]); ("u2", [ "n2" ], [ "n2_y" ]) ];
    tensors = [ ("x", "x_y"); ("w", "w_y"); ("h", "h_y"); ("y", "y_y") ];
    eliminated = [];
    created = [];
    constants = [];
  }

let map_yz =
  {
    F.left = gy;
    right = gz;
    units = [ ("u1", [ "n1_y" ], [ "n1_z" ]); ("u2", [ "n2_y" ], [ "n2_z" ]) ];
    tensors = [ ("x_y", "x_z"); ("w_y", "w_z"); ("h_y", "h_z"); ("y_y", "y_z") ];
    eliminated = [];
    created = [];
    constants = [];
  }

let map_zw =
  {
    F.left = gz;
    right = gw;
    units = [ ("u1", [ "n1_z" ], [ "n1_w" ]); ("u2", [ "n2_z" ], [ "n2_w" ]) ];
    tensors = [ ("x_z", "x_w"); ("w_z", "w_w"); ("h_z", "h_w"); ("y_z", "y_w") ];
    eliminated = [];
    created = [];
    constants = [];
  }

let art m = F.mapping m

let valid m a =
  match F.check m a with
  | Ok s -> s
  | Error e -> Format.kasprintf failwith "%a" pp_plan e

let () = ignore (names, ok_or)
let err_text e = Format.asprintf "%a" Compose_error.pp (Err.Error.kind e)

let compose_ok a b =
  match F.compose a b with Ok m -> m | Error e -> failwith (err_text e)

let local (r : Node_ref.t) = Id.Node_local.to_string r.local_id
let tlocal (r : Tensor_ref.t) = Id.Tensor_local.to_string r.local_id

let show (m : Mapping_artifact.t) =
  List.iter
    (fun (u : Matched_unit.t) ->
      Printf.printf "unit [%s] -> [%s] in [%s] out [%s]\n"
        (String.concat "," (List.map local u.left.nodes))
        (String.concat "," (List.map local u.right.nodes))
        (String.concat "," (List.map tlocal u.right.inputs))
        (String.concat "," (List.map tlocal u.right.outputs)))
    (List.sort
       (fun (a : Matched_unit.t) b ->
         compare (List.map local a.left.nodes) (List.map local b.left.nodes))
       m.units);
  List.iter
    (fun (r : Tensor_relation.t) ->
      Printf.printf "relation %s <-> %s routes %d\n"
        (String.concat "," (List.map tlocal r.left_members))
        (String.concat "," (List.map tlocal r.right_members))
        (List.length r.routes))
    (List.sort
       (fun (a : Tensor_relation.t) b ->
         compare
           (List.map tlocal a.left_members)
           (List.map tlocal b.left_members))
       m.tensor_relations)

module Core = Inference_compare_core
module Ex = Inference_compare_extract
module CF = Coord_fixture
module C = Chain_fixture
module Ops = Inference_compare_compose.Recipe_ops

let planner left right (m : Mapping_artifact.t) =
  Plan.Structural.check ~limits:P.limits ~options:P.options ~left:(F.desc left)
    ~right:(F.desc right) m

let accepted left right m =
  match planner left right m with
  | Ok s -> s
  | Error e -> Format.kasprintf failwith "planner: %a" pp_plan e

let steps_text (r : Tensor_relation.t) =
  List.map
    (fun (v : Value_route.t) ->
      Printf.sprintf "%s->%s:%s"
        (String.concat "," (List.map tlocal v.sources))
        (tlocal v.destination)
        (String.concat ";"
           (List.map
              (fun st -> Wire.canonical_string (Conversion_step.encode st))
              v.steps)))
    r.routes

(* ---- fusion in the first map, decomposition in the second ------------------ *)

let ga =
  F.graph F.key_x
    [ ("n1", "addc:1", [ "x" ], [ "h" ]); ("n2", "addc:2", [ "h" ], [ "y" ]) ]
    ~sources:[ "x" ] ~finals:[ "y" ]

let gb =
  F.graph F.key_y
    [ ("f", "addc:3", [ "x_y" ], [ "y_y" ]) ]
    ~sources:[ "x_y" ] ~finals:[ "y_y" ]

let gc =
  F.graph F.key_z
    [
      ("g1", "addc:1", [ "x_z" ], [ "t_z" ]);
      ("g2", "addc:2", [ "t_z" ], [ "y_z" ]);
    ]
    ~sources:[ "x_z" ] ~finals:[ "y_z" ]

let map_ab =
  {
    F.left = ga;
    right = gb;
    units = [ ("u", [ "n1"; "n2" ], [ "f" ]) ];
    tensors = [ ("x", "x_y"); ("y", "y_y") ];
    eliminated = [];
    created = [];
    constants = [];
  }

let map_bc =
  {
    F.left = gb;
    right = gc;
    units = [ ("u", [ "f" ], [ "g1"; "g2" ]) ];
    tensors = [ ("x_y", "x_z"); ("y_y", "y_z") ];
    eliminated = [];
    created = [];
    constants = [];
  }

let gd =
  F.graph F.key_w
    [ ("h_w", "addc:3", [ "x_w" ], [ "y_w" ]) ]
    ~sources:[ "x_w" ] ~finals:[ "y_w" ]

let map_cd =
  {
    F.left = gc;
    right = gd;
    units = [ ("u", [ "g1"; "g2" ], [ "h_w" ]) ];
    tensors = [ ("x_z", "x_w"); ("y_z", "y_w") ];
    eliminated = [];
    created = [];
    constants = [];
  }

let run_composed ?(donor = Side.Left) left right (m : Mapping_artifact.t) =
  let structural = accepted left right m in
  let extractions =
    Result.get_ok
      (Ex.Extraction.extract_all ~hashers:CF.hashers ~algorithm:CF.algorithm
         structural)
  in
  let bindings = CF.sealed structural () in
  let sealed =
    match
      Core.Sealed_plan.seal ~hashers:CF.hashers ~algorithm:CF.algorithm
        structural extractions bindings
    with
    | Ok p -> p
    | Error e ->
        Format.kasprintf failwith "seal: %a" Core.Sealed_plan.pp_error
          (Err.Error.kind e)
  in
  let chain = C.synthetic [] ~sources:left.F.sources ~finals:[] ~units:[] in
  let t = { CF.structural; extractions; bindings; sealed; chain } in
  let ictx = Codec_fixtures.context () in
  let initial =
    List.map
      (fun s ->
        ( P.ckey s,
          Codec_fixtures.integers ictx C.i64_spec (C.initial_ints chain s) ))
      left.F.sources
  in
  CF.execute t (CF.setup ~donor ~initial_override:initial ())

let summary (r : CF.run) =
  Printf.printf "%s %s %s | %s\n"
    (match r.result.completion.execution with
    | Complete -> "complete"
    | Failed -> "failed"
    | Refused -> "refused"
    | Interrupted -> "interrupted")
    (match r.result.completion.numerical with
    | Agreement -> "agreement"
    | Disagreement -> "disagreement"
    | Not_established -> "not_established")
    (String.concat "," (names r.result.executed))
    (String.concat "," (CF.verdicts r.result))

(* ---- deletion and creation -------------------------------------------------- *)

let dead_graph key suffix extra =
  let nm s = if suffix = "" then s else s ^ "_" ^ suffix in
  F.graph key
    ([ (nm "n1", "addc:1", [ nm "x" ], [ nm "h" ]) ] @ extra nm)
    ~sources:[ nm "x" ]
    ~finals:[ nm "h" ]

let dx =
  dead_graph F.key_x "" (fun nm ->
      [ (nm "nd", "neg", [ nm "x" ], [ nm "dead" ]) ])

let dy =
  dead_graph F.key_y "y" (fun nm ->
      [ (nm "nd", "neg", [ nm "x" ], [ nm "dead" ]) ])

let dz =
  dead_graph F.key_z "z" (fun nm ->
      [ (nm "nc", "neg", [ nm "x" ], [ nm "dead" ]) ])

let map_dxy =
  {
    F.left = dx;
    right = dy;
    units = [ ("u1", [ "n1" ], [ "n1_y" ]); ("u2", [ "nd" ], [ "nd_y" ]) ];
    tensors = [ ("x", "x_y"); ("h", "h_y") ];
    eliminated = [];
    created = [];
    constants = [];
  }

(* The pass deletes the dead node and adds an equally dead one. *)
let map_dyz =
  {
    F.left = dy;
    right = dz;
    units =
      [
        ("u1", [ "n1_y" ], [ "n1_z" ]);
        ("u2", [ "nd_y" ], []);
        ("u3", [], [ "nc_z" ]);
      ];
    tensors = [ ("x_y", "x_z"); ("h_y", "h_z") ];
    eliminated = [ "nd_y" ];
    created = [ "nc_z" ];
    constants = [];
  }

let disposition_text = function
  | Member_disposition.Created { node; _ } -> "created " ^ local node
  | Eliminated { node; _ } -> "eliminated " ^ local node
  | Internal_helper { node; _ } -> "helper " ^ local node
  | Tensor_not_compared { tensor; _ } -> "not compared " ^ tlocal tensor

(* ---- constant folding -------------------------------------------------------- *)

let fx =
  F.graph F.key_x
    [
      ("k1", "addc:1", [ "w" ], [ "wk" ]); ("n", "add", [ "x"; "wk" ], [ "y" ]);
    ]
    ~sources:[ "x"; "w" ] ~finals:[ "y" ]

let fy =
  F.graph F.key_y
    [
      ("k1_y", "addc:1", [ "w_y" ], [ "wk_y" ]);
      ("n_y", "add", [ "x_y"; "wk_y" ], [ "y_y" ]);
    ]
    ~sources:[ "x_y"; "w_y" ] ~finals:[ "y_y" ]

let fz =
  F.graph F.key_z
    [ ("n_z", "add", [ "x_z"; "wf_z" ], [ "y_z" ]) ]
    ~sources:[ "x_z"; "wf_z" ] ~finals:[ "y_z" ]

let map_fxy =
  {
    F.left = fx;
    right = fy;
    units = [ ("u1", [ "k1" ], [ "k1_y" ]); ("u2", [ "n" ], [ "n_y" ]) ];
    tensors = [ ("x", "x_y"); ("w", "w_y"); ("wk", "wk_y"); ("y", "y_y") ];
    eliminated = [];
    created = [];
    constants = [];
  }

(* The pass folds the constant computation into a pinned constant. *)
let map_fyz =
  {
    F.left = fy;
    right = fz;
    units = [ ("u", [ "k1_y"; "n_y" ], [ "n_z" ]) ];
    tensors = [ ("x_y", "x_z"); ("y_y", "y_z") ];
    eliminated = [];
    created = [];
    constants = [ "wf_z" ];
  }

(* ---- identity and association ------------------------------------------------ *)

let rsig (r : Recipe.t) =
  Wire.canonical_string
    (Recipe.encode
       { (Ops.normalize r) with Recipe.id = P.recipe_id "x"; lossless = false })

(* The meaning of a map with every generated identifier erased: units are named
   by the nodes they own, bindings by the unit's nodes and the tensor. *)
let signature (m : Mapping_artifact.t) =
  let nodes (c : Cluster.t) =
    String.concat "," (List.sort compare (List.map local c.nodes))
  in
  let ukey_of = Hashtbl.create 8 in
  List.iter
    (fun (u : Matched_unit.t) ->
      Hashtbl.replace ukey_of
        (Id.Unit.to_string u.unit_id)
        (nodes u.left ^ "|" ^ nodes u.right))
    m.units;
  let uk id = Hashtbl.find ukey_of (Id.Unit.to_string id) in
  let tl l = String.concat "," (List.sort compare (List.map tlocal l)) in
  let steps (v : Value_route.t) =
    String.concat ";"
      (List.map
         (fun s -> Wire.canonical_string (Conversion_step.encode s))
         v.steps)
  in
  List.sort compare
    (List.map
       (fun (u : Matched_unit.t) ->
         Printf.sprintf "unit %s in %s/%s out %s/%s" (uk u.unit_id)
           (tl u.left.inputs) (tl u.right.inputs) (tl u.left.outputs)
           (tl u.right.outputs))
       m.units
    @ List.map
        (fun (r : Tensor_relation.t) ->
          Printf.sprintf "relation %s|%s %s" (tl r.left_members)
            (tl r.right_members)
            (String.concat " "
               (List.sort compare
                  (List.map
                     (fun (v : Value_route.t) ->
                       tl v.sources ^ ">" ^ tlocal v.destination ^ ":" ^ steps v)
                     r.routes))))
        m.tensor_relations
    @ List.map disposition_text m.dispositions
    @ List.map
        (fun (b : Input_binding.t) ->
          Printf.sprintf "binding %s %s %s %s" (uk b.unit)
            (Side.to_string b.side) (tlocal b.tensor) (rsig b.from_common))
        m.frontier.input_bindings
    @ List.map
        (fun (c : Input_check.t) ->
          Printf.sprintf "check %s %s %s %s" (uk c.unit) (Side.to_string c.side)
            (Id.Canonical_value.to_string c.canonical)
            (rsig c.from_imported_inputs))
        m.frontier.input_checks
    @ List.map
        (fun (o : Output_binding.t) ->
          Printf.sprintf "output %s %s %s %s" (uk o.unit)
            (Id.Canonical_value.to_string o.key)
            (rsig o.from_left_outputs)
            (rsig o.from_right_outputs))
        m.frontier.output_bindings
    @ List.map
        (fun (f : Final_output.t) ->
          Printf.sprintf "final %s %s %s"
            (Id.Canonical_value.to_string f.canonical)
            (tlocal f.left) (tlocal f.right))
        m.frontier.final_outputs
    @ List.map
        (fun (c : Canonical_spec.t) ->
          "canonical " ^ Id.Canonical_value.to_string c.key)
        m.frontier.canonical_values
    @ List.map
        (fun (i : Initial_binding.t) ->
          "initial " ^ Id.Canonical_value.to_string i.key)
        m.frontier.initial_values)

let identity_map g =
  {
    F.left = g;
    right = g;
    units =
      List.map
        (fun (name, op, _, _) ->
          ("u_" ^ name, [ name ], [ name ]) |> fun x ->
          ignore op;
          x)
        g.F.nodes;
    tensors =
      List.sort_uniq compare
        (List.concat_map
           (fun (_, _, o, r) -> List.map (fun t -> (t, t)) (o @ r))
           g.F.nodes);
    eliminated = [];
    created = [];
    constants = [];
  }

let same_meaning what a b =
  let sa = signature a and sb = signature b in
  if sa = sb then
    Printf.printf "%s: same meaning (%d facts)\n" what (List.length sa)
  else (
    Printf.printf "%s: DIFFERENT\n" what;
    List.iter
      (fun l -> if not (List.mem l sb) then print_endline ("  only left: " ^ l))
      sa;
    List.iter
      (fun l ->
        if not (List.mem l sa) then print_endline ("  only right: " ^ l))
      sb)

(* ---- refusals ------------------------------------------------------------------ *)

let refused a b =
  match F.compose a b with
  | Ok _ -> print_endline "composed"
  | Error e -> print_endline (err_text e)

(* ---- comparison pairs ------------------------------------------------------------ *)

let with_pair (m : Mapping_artifact.t) ~unit ~id ~left ~right ~left_steps
    ~right_steps ~spec =
  let pair =
    {
      Member_pair.pair_id = P.text (Id.Member_pair.of_string id);
      comparison_id = P.comparison ("cmp_" ^ id);
      left;
      right;
      claim = Identical;
      left_to_comparison = left_steps;
      right_to_comparison = right_steps;
      comparison_spec = spec;
    }
  in
  {
    m with
    units =
      List.map
        (fun (u : Matched_unit.t) ->
          if Id.Unit.to_string u.unit_id = unit then
            { u with output_pairs = pair.pair_id :: u.output_pairs }
          else u)
        m.units;
    tensor_relations =
      List.map
        (fun (r : Tensor_relation.t) ->
          if
            List.exists (Tensor_ref.equal left) r.left_members
            && List.exists (Tensor_ref.equal right) r.right_members
          then { r with member_pairs = pair :: r.member_pairs }
          else r)
        m.tensor_relations;
  }

let spec_of g = F.spec_of g

(* ---- claims do not strengthen ---------------------------------------------------- *)

let with_claim (m : Mapping_artifact.t) claim =
  {
    m with
    tensor_relations =
      List.map
        (fun (r : Tensor_relation.t) ->
          {
            r with
            routes =
              List.map (fun (v : Value_route.t) -> { v with claim }) r.routes;
          })
        m.tensor_relations;
  }

let claim_text = function
  | Claim.Approximate ps ->
      "approximate " ^ String.concat "," (List.map Id.Precision.to_string ps)
  | Equivalent -> "equivalent"
  | Identical -> "identical"
  | Unverifiable -> "unverifiable"
