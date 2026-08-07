(* Project the EXPORTED PROGRAM's own graph — the source view.

   Successful decoding always yields one, which is what makes [Graph_stage
   Source] the single unconditional row of §5.4's matrix: it claims that the
   model.json parsed, and nothing more. Every stage past it is conditional on
   lowering, and lowering is where operator support enters.

   Deliberately NOT [Me_build.Make]. That functor is parameterised over a
   dialect whose operands are [Graph_ir.Tensor_id.t] and whose hierarchy is a
   [Graph_ir.Group.t] tree; this side has neither. Its values are SSA NAMES and
   its hierarchy is [nn_module_stack], so sharing the body would mean
   generalising over both, and the two projections would still have to be read
   together to be understood.

   ROOT-SCOPED, like [Me_pt2]. The node ids here are exactly the ones the
   sidecar's origins render to, which is what lets the import comparison pair
   the two panes; a nested graph has no signature of its own and no native
   counterpart to pair with.

   See .ai/model_explorer_design.md. *)

type error =
  [ Me_ids.error
  | `Over_limit of string * int
  | `Unknown_producer of string  (** an SSA name nothing in the graph defines *)
  ]

val pp_error : Format.formatter -> [< error ] -> unit

val module_levels : string -> string list
(** The namespace levels of an [nn_module_stack] value: [";"]-separated entries
    of [path,name,class], outermost first.

    Two rules, both of which a test pins because neither is visible in a
    rendering that happens not to exercise it. The outermost entry names the
    whole module and carries an EMPTY name, so it contributes no level — exactly
    as [Me_build] drops the root group's own component. And each [name] is the
    full dotted path, so a level is taken RELATIVE to its parent
    ([layer1 / 0 / conv1], not [layer1 / layer1.0 / layer1.0.conv1]); a level
    that does not extend its parent is kept whole, since dropping a prefix that
    is not there would be guessing at a hierarchy the exporter did not write. *)

val node_ids : Pytorch_types.Graph.t -> string list
(** The root graph's node ids, in graph order — [Me_pt2]'s [~source_nodes].
    Total and cheap: it is the left-pane universe, and deriving it from the
    projection would make a mapping depend on a projection succeeding. *)

val graph :
  limits:Me_limits.Limits.t ->
  Pytorch_types.GraphModule.t ->
  (Model_explorer.Graph.t, [> error ]) Core.result
(** One [GraphNode] per PT2 node, plus one boundary node per graph input and
    output. The id is [Me_ids.pt2_graph Graph_path.root].

    [GraphModule.t] rather than [Graph.t] because the INPUT KIND lives in
    [signature]: a parameter, buffer or tensor constant becomes [const:], a user
    input [in:]. Guessing it from the exporter's [p_]/[b_]/[c_] name prefixes
    would be inventing a rule the schema already states.

    Edges follow TENSOR-VALUED arguments only. A [sym_int] operand names an
    entry in [sym_int_values], not a node output, and drawing an edge for it
    would assert a dataflow dependency the graph does not have.

    An SSA name that no node output and no graph input defines is
    [`Unknown_producer] rather than a dropped edge — a silently missing edge is
    a picture that is wrong rather than one that is incomplete. *)
