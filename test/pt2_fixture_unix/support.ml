module Fx = Pt2_checkpoint_map_test.Fixtures
module U = Pt2_fixture_unix

let write path data =
  Out_channel.with_open_bin path (fun oc -> output_string oc data)

let read path = In_channel.with_open_bin path In_channel.input_all

let rec remove_tree path =
  match (Unix.lstat path).Unix.st_kind with
  | exception Unix.Unix_error _ -> ()
  | Unix.S_DIR ->
      Array.iter
        (fun n -> remove_tree (Filename.concat path n))
        (Sys.readdir path);
      Unix.rmdir path
  | _ -> Sys.remove path

let with_dir f =
  let dir = Filename.temp_file "pt2_fixture" "" in
  Sys.remove dir;
  Unix.mkdir dir 0o700;
  Fun.protect ~finally:(fun () -> remove_tree dir) (fun () -> f dir)

let unstyle = Pt2_checkpoint_map_test.Map_test.unstyle

let show_error e =
  let text = unstyle (Fmt.str "%a" U.Fixture.pp_error e) in
  (* Paths of the temporary cache vary per run; drop everything up to the
     file name. *)
  let text =
    match String.index_opt text '\n' with
    | Some i -> String.sub text 0 i
    | None -> text
  in
  print_endline text

let report = function Ok _ -> print_endline "ok" | Error e -> show_error e

(* A transport serving [Pt2_fixture_test.World.served]. [tweak] rewrites what a URL returns;
   [fail] makes a URL error; [calls] records every request in order. *)
let transport ?(tweak = fun _ b -> b) ?(fail = fun _ -> None) calls :
    U.Transport.t =
 fun ~url ~dest ->
  calls := url :: !calls;
  match fail url with
  | Some m ->
      write dest "partial";
      Error m
  | None -> (
      match List.assoc_opt url Pt2_fixture_test.World.served with
      | None -> Error "404"
      | Some body ->
          write dest (tweak url body);
          Ok ())

let cache dir =
  match Err.payload (U.Cache.create (Filename.concat dir "cache")) with
  | Ok c -> c
  | Error _ -> failwith "cache"

let names dir = List.sort String.compare (Array.to_list (Sys.readdir dir))
