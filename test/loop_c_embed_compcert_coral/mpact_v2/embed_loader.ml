include Loop_c_embed_compcert_coral.Compcert_coral_loader.Make (struct
  let dir = Sys.getenv "CORALNPU_DATA"
  let configuration = "mpact-v2"
  let deadline = 120.
end)
