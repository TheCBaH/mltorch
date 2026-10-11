module W = Wire

type scope = Endpoint_graph | Within_cluster
type boundary = Cluster_input | Cluster_output | Graph_output

module Node_pattern = struct
  type t = { node : Node_ref.t option; operator : Operator_ref.t option }

  let decode : t W.decoder =
    W.record (fun o ->
        let ( let* ) = W.( let* ) in
        let* node = W.Obj.optional o "node" Node_ref.decode in
        let* operator = W.Obj.optional o "operator" Operator_ref.decode in
        Ok { node; operator })

  let encode p =
    W.obj
      [
        W.field "node" (W.enc_option Node_ref.encode) p.node;
        W.field "operator" (W.enc_option Operator_ref.encode) p.operator;
      ]
end

module Producer = struct
  type t = { node : Node_pattern.t; port : Output_port.t option }

  let decode : t W.decoder =
    W.record (fun o ->
        let ( let* ) = W.( let* ) in
        let* node = W.Obj.required o "node" Node_pattern.decode in
        let* port = W.Obj.optional o "port" Output_port.decode in
        Ok { node; port })

  let encode p =
    W.obj
      [
        W.field "node" Node_pattern.encode p.node;
        W.field "port" (W.enc_option Output_port.encode) p.port;
      ]
end

module Consumer = struct
  type t = { node : Node_pattern.t; port : Input_port.t option }

  let decode : t W.decoder =
    W.record (fun o ->
        let ( let* ) = W.( let* ) in
        let* node = W.Obj.required o "node" Node_pattern.decode in
        let* port = W.Obj.optional o "port" Input_port.decode in
        Ok { node; port })

  let encode c =
    W.obj
      [
        W.field "node" Node_pattern.encode c.node;
        W.field "port" (W.enc_option Input_port.encode) c.port;
      ]
end

module Hop = struct
  type t = {
    node : Node_pattern.t;
    input_port : Input_port.t option;
    output_port : Output_port.t option;
  }

  let decode : t W.decoder =
    W.record (fun o ->
        let ( let* ) = W.( let* ) in
        let* node = W.Obj.required o "node" Node_pattern.decode in
        let* input_port = W.Obj.optional o "input_port" Input_port.decode in
        let* output_port = W.Obj.optional o "output_port" Output_port.decode in
        Ok { node; input_port; output_port })

  let encode h =
    W.obj
      [
        W.field "node" Node_pattern.encode h.node;
        W.field "input_port" (W.enc_option Input_port.encode) h.input_port;
        W.field "output_port" (W.enc_option Output_port.encode) h.output_port;
      ]
end

module Count_range = struct
  type t = { minimum : int64; maximum : int64 option }

  let decode : t W.decoder =
    W.record (fun o ->
        let ( let* ) = W.( let* ) in
        let* minimum = W.Obj.required o "minimum" W.nonneg_int64 in
        let* maximum = W.Obj.optional o "maximum" W.nonneg_int64 in
        match maximum with
        | Some m when Int64.compare m minimum < 0 ->
            W.fail (W.Constraint Out_of_range)
        | _ -> Ok { minimum; maximum })

  let encode r =
    W.obj
      [
        W.field "minimum" W.enc_int64 r.minimum;
        W.field "maximum" (W.enc_option W.enc_int64) r.maximum;
      ]
end

module Path = struct
  type t = { scope : scope; through : Hop.t list; sink : Consumer.t }
end

type t =
  | All_of of t list
  | Any_of of t list
  | Any_tensor
  | Boundary of boundary
  | Consumed_by of { scope : scope; consumer : Consumer.t }
  | Fan_out of { scope : scope; count : Count_range.t }
  | Flows_to of Path.t
  | Not of t
  | Produced_by of { scope : scope; producer : Producer.t }
  | Tensor of Tensor_ref.t

let decode_scope, encode_scope =
  W.enum ~kind:"scope"
    [ (Endpoint_graph, "endpoint_graph"); (Within_cluster, "within_cluster") ]

let decode_boundary, encode_boundary =
  W.enum ~kind:"boundary"
    [
      (Cluster_input, "cluster_input");
      (Cluster_output, "cluster_output");
      (Graph_output, "graph_output");
    ]

let decode_path : Path.t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* scope = W.Obj.required o "scope" decode_scope in
      let* through = W.Obj.required o "through" (W.list Hop.decode) in
      let* sink = W.Obj.required o "sink" Consumer.decode in
      Ok { Path.scope; through; sink })

let encode_path (p : Path.t) =
  W.obj
    [
      W.field "scope" encode_scope p.scope;
      W.field "through" (W.enc_list Hop.encode) p.through;
      W.field "sink" Consumer.encode p.sink;
    ]

(* Empty [All_of]/[Any_of] lists are rejected here, not at planning: an empty
   conjunction would otherwise be a universal policy and an empty disjunction
   a silent never-match. *)
let nonempty items =
  if items = [] then W.fail (W.Constraint Empty_list) else Ok items

let rec decode : t W.decoder =
 fun j ->
  W.variant
    (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "all_of" ->
          let* items = W.Obj.required o "items" (W.list decode) in
          let* items = nonempty items in
          Ok (All_of items)
      | "any_of" ->
          let* items = W.Obj.required o "items" (W.list decode) in
          let* items = nonempty items in
          Ok (Any_of items)
      | "any_tensor" -> Ok Any_tensor
      | "boundary" ->
          let* b = W.Obj.required o "boundary" decode_boundary in
          Ok (Boundary b)
      | "consumed_by" ->
          let* scope = W.Obj.required o "scope" decode_scope in
          let* consumer = W.Obj.required o "consumer" Consumer.decode in
          Ok (Consumed_by { scope; consumer })
      | "fan_out" ->
          let* scope = W.Obj.required o "scope" decode_scope in
          let* count = W.Obj.required o "count" Count_range.decode in
          Ok (Fan_out { scope; count })
      | "flows_to" ->
          let* p = W.Obj.required o "path" decode_path in
          Ok (Flows_to p)
      | "not" ->
          let* t = W.Obj.required o "operand" decode in
          Ok (Not t)
      | "produced_by" ->
          let* scope = W.Obj.required o "scope" decode_scope in
          let* producer = W.Obj.required o "producer" Producer.decode in
          Ok (Produced_by { scope; producer })
      | "tensor" ->
          let* r = W.Obj.required o "tensor" Tensor_ref.decode in
          Ok (Tensor r)
      | tag -> W.unknown_tag ~kind:"tensor_selector" tag)
    j

let rec encode = function
  | All_of items ->
      W.tagged "all_of" [ W.field "items" (W.enc_list encode) items ]
  | Any_of items ->
      W.tagged "any_of" [ W.field "items" (W.enc_list encode) items ]
  | Any_tensor -> W.tagged "any_tensor" []
  | Boundary b -> W.tagged "boundary" [ W.field "boundary" encode_boundary b ]
  | Consumed_by { scope; consumer } ->
      W.tagged "consumed_by"
        [
          W.field "scope" encode_scope scope;
          W.field "consumer" Consumer.encode consumer;
        ]
  | Fan_out { scope; count } ->
      W.tagged "fan_out"
        [
          W.field "scope" encode_scope scope;
          W.field "count" Count_range.encode count;
        ]
  | Flows_to p -> W.tagged "flows_to" [ W.field "path" encode_path p ]
  | Not t -> W.tagged "not" [ W.field "operand" encode t ]
  | Produced_by { scope; producer } ->
      W.tagged "produced_by"
        [
          W.field "scope" encode_scope scope;
          W.field "producer" Producer.encode producer;
        ]
  | Tensor r -> W.tagged "tensor" [ W.field "tensor" Tensor_ref.encode r ]
