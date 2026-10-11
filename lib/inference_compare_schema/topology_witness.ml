module W = Wire

module Use = struct
  type t = { tensor : Tensor_ref.t; node : Node_ref.t; port : Input_port.t }

  let decode : t W.decoder =
    W.record (fun o ->
        let ( let* ) = W.( let* ) in
        let* tensor = W.Obj.required o "tensor" Tensor_ref.decode in
        let* node = W.Obj.required o "node" Node_ref.decode in
        let* port = W.Obj.required o "port" Input_port.decode in
        Ok { tensor; node; port })

  let encode u =
    W.obj
      [
        W.field "tensor" Tensor_ref.encode u.tensor;
        W.field "node" Node_ref.encode u.node;
        W.field "port" Input_port.encode u.port;
      ]
end

module Hop = struct
  type t = { input : Use.t; output : Tensor_ref.t; port : Output_port.t }

  let decode : t W.decoder =
    W.record (fun o ->
        let ( let* ) = W.( let* ) in
        let* input = W.Obj.required o "input" Use.decode in
        let* output = W.Obj.required o "output" Tensor_ref.decode in
        let* port = W.Obj.required o "port" Output_port.decode in
        Ok { input; output; port })

  let encode h =
    W.obj
      [
        W.field "input" Use.encode h.input;
        W.field "output" Tensor_ref.encode h.output;
        W.field "port" Output_port.encode h.port;
      ]
end

module Path = struct
  type t = { through : Hop.t list; sink : Use.t }

  let decode : t W.decoder =
    W.record (fun o ->
        let ( let* ) = W.( let* ) in
        let* through = W.Obj.required o "through" (W.list Hop.decode) in
        let* sink = W.Obj.required o "sink" Use.decode in
        Ok { through; sink })

  let encode p =
    W.obj
      [
        W.field "through" (W.enc_list Hop.encode) p.through;
        W.field "sink" Use.encode p.sink;
      ]
end

type t = {
  side : Side.t;
  anchor : Tensor_ref.t;
  producers : (Node_ref.t * Output_port.t) list;
  consumers : Use.t list;
  paths : Path.t list;
  fan_out_counts : (Tensor_selector.scope * int64) list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* side = W.Obj.required o "side" Side.decode in
      let* anchor = W.Obj.required o "anchor" Tensor_ref.decode in
      let* producers =
        W.Obj.required o "producers"
          (W.list (W.pair Node_ref.decode Output_port.decode))
      in
      let* consumers = W.Obj.required o "consumers" (W.list Use.decode) in
      let* paths = W.Obj.required o "paths" (W.list Path.decode) in
      let* fan_out_counts =
        W.Obj.required o "fan_out_counts"
          (W.list (W.pair Tensor_selector.decode_scope W.nonneg_int64))
      in
      Ok { side; anchor; producers; consumers; paths; fan_out_counts })

let encode w =
  W.obj
    [
      W.field "side" Side.encode w.side;
      W.field "anchor" Tensor_ref.encode w.anchor;
      W.field "producers"
        (W.enc_list (W.enc_pair Node_ref.encode Output_port.encode))
        w.producers;
      W.field "consumers" (W.enc_list Use.encode) w.consumers;
      W.field "paths" (W.enc_list Path.encode) w.paths;
      W.field "fan_out_counts"
        (W.enc_list (W.enc_pair Tensor_selector.encode_scope W.enc_int64))
        w.fan_out_counts;
    ]
