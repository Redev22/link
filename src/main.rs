use whisper_rs::{WhisperContext, FullParams, SamplingStrategy};
use std::{time, env};
use sfml::audio::{capture, SoundBufferRecorder};

const THREADS: i32 = 8;
const MODEL: &str = "medium"; // tested whisper models are medium and small
const LANGUAGE: &str = "it"; // set language for whisper
const REC_TIME: u64 = 8; // how many seconds the program listens for speech
const DEBUG_LEVEL: i32 = 1;
fn main() {
    //model should be placed inside a folder {LINK_MOD_PATH}/whisper-{model}/ggml-model.bin
    let mod_path = match env::var("LINK_MOD_PATH"){
        Ok(path) => format!("{}{}{}{}", path, "whisper-", MODEL, "/ggml-model.bin"),
        Err(e) => panic!("You must set env LINK_MOD_PATH to be the folder containing the models in ggml format (see convert.py): {:?}", e)
    };
    // load a context and model
    let mut ctx = WhisperContext::new(mod_path.as_str()).expect("failed to load model");
    
    // create a params object
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    //other options
    params.set_n_threads(THREADS);
    params.set_translate(false);
    params.set_language(Some(LANGUAGE));
    params.set_print_progress(false);
    /* 
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    */

    // record_audio handles recording and outputs in the correct format (16KHz sampling rate, mono, 32 bit floats)
    let audio_data = record_audio(time::Duration::from_secs(REC_TIME));


    // run the model
    ctx.full(params, &audio_data[..])
        .expect("failed to run model");

    // fetch the results
    let num_segments = ctx.full_n_segments();
    for i in 0..num_segments {
        let segment = ctx.full_get_segment_text(i).expect("failed to get segment");
        let start_timestamp = ctx.full_get_segment_t0(i);
        let end_timestamp = ctx.full_get_segment_t1(i);
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }

}



// Records audio samples from the microphone for the given duration, and
// returns them as the correct format
fn record_audio(time: time::Duration) -> Vec<f32> {
    // Check that the device can capture audio
    assert!(
        capture::is_available(),
        "Sorry, audio capture is not supported by your system"
    );

    let sample_rate: u32 = 16000; // Whisper needs a sample rate of 16KHz
    let mut recorder = SoundBufferRecorder::new();

    // Audio capture is done in a separate thread,
    // so we can block the main thread while it is capturing
    println!("Recording...");
    recorder.start(sample_rate);
    std::thread::sleep(time);
    println!("Finished recording");
    recorder.stop();

    // Get the buffer containing the captured data
    let buffer = recorder.buffer();

    // Display captured sound information
    if DEBUG_LEVEL >= 1 {
        println!("Sound information :");
        println!(" {} seconds", buffer.duration().as_seconds());
        println!(" {} samples / sec", buffer.sample_rate());
        println!(" {} channels", buffer.channel_count());
    }
    let mut samples = vec!();
    samples.extend_from_slice(buffer.samples());

    whisper_rs::convert_integer_to_float_audio(&samples)
}