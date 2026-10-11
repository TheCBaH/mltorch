open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Ex = Inference_compare_extract

type endpoint =
  | Endpoint : {
      runner :
        (module Runner.GRAPH_RUNNER with type Graph.t = 'g and type Error.t = 'e);
      build : Ex.Extraction.t -> ('g, Ex.Extract_error.t) Err.t;
    }
      -> endpoint

type error = { phase : Phase.t; code : Failure.code; text : string }

type imported = {
  readback : (Tensor_ref.t * L.t) list;
  run : unit -> ((Tensor_ref.t * L.t) list, error) result;
  release : unit -> unit;
}

type t = {
  identity : Engine_identity.t;
  prepare : Ex.Extraction.t -> (unit, error) result;
  import : Id.Unit.t -> (Tensor_ref.t * L.t) list -> (imported, error) result;
  close : unit -> unit;
}

let make ctx (ep : endpoint) : t =
  match ep with
  | Endpoint { runner = (module R); build } ->
      let prepared : (Id.Unit.t * (R.Prepared.t * Ex.Extraction.t)) list ref =
        ref []
      in
      let closed = ref false in
      let render e = Format.asprintf "%a" R.Error.pp (Err.Error.kind e) in
      let err phase code text = Error { phase; code; text } in
      (* Exhaustion keeps its own code whatever phase it struck in. *)
      let eng phase code e =
        let code =
          if R.Error.is_resource_exhaustion (Err.Error.kind e) then
            Failure.Resource_exhausted
          else code
        in
        err phase code (render e)
      in
      let prepare (ex : Ex.Extraction.t) =
        match build ex with
        | Error e ->
            err Preparation Preparation_refused
              (Format.asprintf "%a" Ex.Extract_error.pp (Err.Error.kind e))
        | Ok graph -> (
            match R.prepare graph with
            | Error e -> eng Preparation Preparation_refused e
            | Ok p ->
                prepared := (ex.unit, (p, ex)) :: !prepared;
                Ok ())
      in
      let close () =
        if not !closed then (
          closed := true;
          List.iter (fun (_, (p, _)) -> R.close p) !prepared;
          prepared := [])
      in
      let import unit inputs =
        match List.find_opt (fun (u, _) -> Id.Unit.equal u unit) !prepared with
        | None -> err Import Preparation_refused "unit was not prepared"
        | Some (_, (p, ex)) -> (
            let handles = ref [] in
            let release () =
              let hs = !handles in
              handles := [];
              List.iter (fun (_, h) -> R.release_value h) hs
            in
            let spec_of t =
              List.find_map
                (fun (s : Graph_source.t) ->
                  if Tensor_ref.equal s.tensor t then Some s.spec else None)
                ex.body.sources
            in
            let rec import_all = function
              | [] -> Ok ()
              | (t, v) :: rest -> (
                  match spec_of t with
                  | None ->
                      err Import Input_transfer_failed
                        "input is not a source of the graph"
                  | Some spec -> (
                      match R.import ctx p spec v with
                      | Error e -> eng Import Input_transfer_failed e
                      | Ok h ->
                          handles := (t, h) :: !handles;
                          import_all rest))
            in
            match import_all inputs with
            | Error e ->
                release ();
                Error e
            | Ok () -> (
                let ordered = List.rev !handles in
                let rec back acc = function
                  | [] -> Ok (List.rev acc)
                  | (t, h) :: rest -> (
                      match R.export ctx h with
                      | Ok v -> back ((t, v) :: acc) rest
                      | Error e ->
                          List.iter (fun (_, v) -> L.drop v) acc;
                          eng Input_check Input_transfer_failed e)
                in
                match back [] ordered with
                | Error e ->
                    release ();
                    Error e
                | Ok readback ->
                    let run () =
                      match R.run p ~inputs:ordered with
                      | Error e -> eng Execution Execution_failed e
                      | Ok outs ->
                          let release_outs () =
                            List.iter (fun (_, h) -> R.release_value h) outs
                          in
                          let rec snapshot acc = function
                            | [] -> Ok (List.rev acc)
                            | (t, h) :: rest -> (
                                match R.export ctx h with
                                | Ok v -> snapshot ((t, v) :: acc) rest
                                | Error e ->
                                    List.iter (fun (_, v) -> L.drop v) acc;
                                    eng Export Output_inventory_failed e)
                          in
                          let r = snapshot [] outs in
                          release_outs ();
                          r
                    in
                    Ok { readback; run; release }))
      in
      { identity = R.identity (); prepare; import; close }
