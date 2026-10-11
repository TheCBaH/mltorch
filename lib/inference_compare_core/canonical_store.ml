open Inference_compare_schema
module L = Inference_compare_codecs.Logical_value

type entry = { value : L.t; instance : Id.Value_instance.t; mutable uses : int }

module Key_map = Map.Make (struct
  type t = Id.Canonical_value.t

  let compare = Id.Canonical_value.compare
end)

type t = { mutable entries : entry Key_map.t }

let create () = { entries = Key_map.empty }

let add t key ~instance ~uses value =
  if Key_map.mem key t.entries then Error `Already_present
  else if uses <= 0 then (
    L.drop value;
    Ok ())
  else (
    t.entries <- Key_map.add key { value; instance; uses } t.entries;
    Ok ())

let get t key = Option.map (fun e -> e.value) (Key_map.find_opt key t.entries)

let instance t key =
  Option.map (fun e -> e.instance) (Key_map.find_opt key t.entries)

let uses_left t key =
  Option.map (fun e -> e.uses) (Key_map.find_opt key t.entries)

let finish_uses t key n =
  match Key_map.find_opt key t.entries with
  | None -> invalid_arg "Canonical_store.finish_uses: unknown value"
  | Some e ->
      if n > e.uses then
        invalid_arg "Canonical_store.finish_uses: more reads than counted";
      e.uses <- e.uses - n;
      if e.uses = 0 then (
        L.drop e.value;
        t.entries <- Key_map.remove key t.entries)

let live t = List.map fst (Key_map.bindings t.entries)

let clear t =
  Key_map.iter (fun _ e -> L.drop e.value) t.entries;
  t.entries <- Key_map.empty
