(* The wire shape of a pinned asset, shared by the cohort manifest, the
   publication index and the per-artifact manifest. Closed schemas elsewhere,
   but a pinned asset may grow members, so unknown ones are skipped. *)

module Document = Pt2_checkpoint_map.Document

type t = { name : string; sha256 : string; size : int64; url : string }

let jsont =
  Jsont.Object.map ~kind:"pinned asset" (fun name sha256 size url ->
      { name; sha256; size; url })
  |> Jsont.Object.mem "name" Jsont.string
  |> Jsont.Object.mem "sha256" Jsont.string
  |> Jsont.Object.mem "size" Jsont.int64
  |> Jsont.Object.mem "url" Jsont.string
  |> Jsont.Object.skip_unknown |> Jsont.Object.finish

let to_pin (w : t) =
  Document.pin_of_fields ~name:w.name ~sha256:w.sha256 ~size:w.size ~url:w.url

(* A pinned file with no name of its own: the publication index is addressed by
   its URL, so its file name is the URL's last segment. *)
type anonymous = { a_sha256 : string; a_size : int64; a_url : string }

let anonymous_jsont =
  Jsont.Object.map ~kind:"pinned file" (fun a_sha256 a_size a_url ->
      { a_sha256; a_size; a_url })
  |> Jsont.Object.mem "sha256" Jsont.string
  |> Jsont.Object.mem "size" Jsont.int64
  |> Jsont.Object.mem "url" Jsont.string
  |> Jsont.Object.skip_unknown |> Jsont.Object.finish

let last_segment url =
  match String.rindex_opt url '/' with
  | Some i -> String.sub url (i + 1) (String.length url - i - 1)
  | None -> url

let anonymous_to_pin a =
  Document.pin_of_fields ~name:(last_segment a.a_url) ~sha256:a.a_sha256
    ~size:a.a_size ~url:a.a_url

let equal (a : Document.Pin.t) (b : Document.Pin.t) =
  String.equal a.name b.name
  && Pt2_sha256.Digest.equal a.sha256 b.sha256
  && Int64.equal a.size b.size && String.equal a.url b.url
