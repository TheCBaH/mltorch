(* Stage 0A contract spike: the fixture.

   Hand-authored, and deliberately NOT produced by an exporter -- there is no
   exporter yet, and the questions this spike settles are about the RENDERER,
   not about our projection of a model. What it does use is the real
   [Model_explorer] binding, so the JSON below is the JSON the export path will
   emit rather than a second description of the same schema.

   Every shape here exists to answer one question the coordinator design
   currently assumes. See .ai/model_explorer_design.md. *)

module ME = Model_explorer

let node ?(namespace = "") ?(incoming = []) ?(subgraphs = []) ?(outputs = 1) id
    label =
  ME.GraphNode.create ~id ~label ~namespace
    ~incomingEdges:
      (List.map
         (fun (src, slot) ->
           ME.IncomingEdge.create ~sourceNodeId:src ~sourceNodeOutputId:slot
             ~targetNodeInputId:"0" ())
         incoming)
    ~outputsMetadata:
      (List.init outputs (fun i ->
           ME.MetadataItem.create ~id:(string_of_int i) ~attrs:[]))
    ~subgraphIds:subgraphs ()

(* --- the two panes of a comparison ---

   Left has four nodes, right has three: one PT2-ish node decomposed into two,
   two folded into one, one created and one deleted. That is the whole shape
   sync navigation has to express, in the smallest graph that expresses it. *)

let left =
  ME.Graph.create ~id:"g/native/000"
    ~nodes:
      [
        node "n0" "input" ~namespace:"";
        node "n1" "conv" ~namespace:"features" ~incoming:[ ("n0", "0") ];
        node "n2" "bias" ~namespace:"features" ~incoming:[ ("n1", "0") ];
        node "n3" "relu" ~namespace:"features" ~incoming:[ ("n2", "0") ];
      ]
    ()

let right =
  ME.Graph.create ~id:"g/native/003"
    ~nodes:
      [
        node "m0" "input" ~namespace:"";
        (* n1 and n2 folded *)
        node "m1" "conv_bias" ~namespace:"features" ~incoming:[ ("m0", "0") ];
        (* created: no counterpart on the left *)
        node "m2" "layout" ~namespace:"features" ~incoming:[ ("m1", "0") ];
      ]
    ()

(* --- a graph with a subgraph link, installed LATE ---

   The parent ships with no [subgraphIds] on [n3]. The spike installs the
   child graph and the link AFTER [modelGraphProcessed] has fired, then enters
   it and returns -- which is the question the detail path depends on and
   which nothing in the static fixture can answer. *)

let child =
  ME.Graph.create ~id:"expr/g/native/000/n3/t0"
    ~nodes:[ node "e0" "add"; node "e1" "mul" ~incoming:[ ("e0", "0") ] ]
    ()

(* --- the bipartite flow ---

   One node per state and one per transition, edges state -> transition ->
   state. Selecting a transition node is how a comparison is opened, and an
   edge cannot carry that because Model Explorer has no edge event. *)

let flow =
  ME.Graph.create ~id:"g/flow"
    ~nodes:
      [
        node "s/pt2/000" "PT2" ~subgraphs:[ "g/native/000" ];
        node "t/native/000" "import" ~incoming:[ ("s/pt2/000", "0") ];
        node "s/native/000" "initial native"
          ~incoming:[ ("t/native/000", "0") ]
          ~subgraphs:[ "g/native/000" ];
        node "t/native/001" "fold" ~incoming:[ ("s/native/000", "0") ];
        node "s/native/003" "canonical"
          ~incoming:[ ("t/native/001", "0") ]
          ~subgraphs:[ "g/native/003" ];
      ]
    ()

let collection =
  ME.GraphCollection.create ~label:"mltorch:spike" ~graphs:[ flow; left; right ]
    ()

(* The child is a SEPARATE collection payload, written to its own file: the
   spike installs it at runtime, which is the point. *)
let detail_collection =
  ME.GraphCollection.create ~label:"mltorch:spike" ~graphs:[ child ] ()

let write path jsont v =
  match Jsont_bytesrw.encode_string ~format:Jsont.Indent jsont v with
  | Error e -> failwith (path ^ ": " ^ e)
  | Ok s ->
      let oc = open_out path in
      output_string oc s;
      close_out oc

let () =
  let dir = if Array.length Sys.argv > 1 then Sys.argv.(1) else "." in
  write
    (Filename.concat dir "collection.json")
    (Jsont.list ME.GraphCollection.jsont)
    [ collection ];
  write
    (Filename.concat dir "detail.json")
    (Jsont.list ME.GraphCollection.jsont)
    [ detail_collection ];
  print_endline "wrote collection.json detail.json"
