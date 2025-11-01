function safeClone(obj) {
    return typeof structuredClone === 'function'
        ? structuredClone(obj)
        : JSON.parse(JSON.stringify(obj))
}

function toCommand(data_raw, id) {
    const data = safeClone(data_raw)
    if (!data || typeof data !== 'object') throw new Error('toCommand: invalid data')

    if (
        data.type === 'prepare_star' &&
        data.hasOwnProperty('even_odd_input') &&
        data.even_odd_input
    )
        data.full = 'None'
    if (
        data.type === 'prepare_star' &&
        data.hasOwnProperty('even_odd_input') &&
        !data.even_odd_input
    ) {
        data.odd = 'None'
        data.evn = 'None'
    }
    if (data.type !== 'prepare_star' && data.type !== 'star2json') {
        data.output_dir = data.type + '/job' + id + '_' + data.name
    }

    //     CTF_mode: 'None',
    // isCTFflipped: false,
    // do_phaseflip_input: true,
    // clip_first_peak_mode: 1,
    // bfactor: 0,

    // noise_level: 0,
    // noise_mode: 'nofilter',

    // with_predict: true,
    // pred_tomo_idx: 1,

    // even_odd_input: true,
    // snrfalloff: 0,
    // deconvstrength: 1,
    // highpassnyquist: 0.02
    if (data.hasOwnProperty('CTF_mode')) {
        if (data.CTF_mode === 'None') {
            delete data.do_phaseflip_input
            delete data.clip_first_peak_mode
            delete data.bfactor
            delete data.snrfalloff
            delete data.deconvstrength
            delete data.highpassnyquist
        } else if (data.CTF_mode === 'network') {
            delete data.snrfalloff
            delete data.deconvstrength
            delete data.highpassnyquist
        } else if (data.CTF_mode === 'phase_only') {
            delete data.clip_first_peak_mode
            delete data.bfactor
            delete data.snrfalloff
            delete data.deconvstrength
            delete data.highpassnyquist
        } else if (data.CTF_mode === 'wiener') {
            delete data.do_phaseflip_input
            delete data.snrfalloff
            delete data.deconvstrength
            delete data.highpassnyquist
        }
    }
    if (data.hasOwnProperty('even_odd_input') && data.type === 'refine' && data.even_odd_input) {
        delete data.input_column
        delete data.noise_level
        delete data.noise_mode
    }

    let result = ''
    const pyBoolMap = { true: 'True', false: 'False', null: 'None' }
    for (const key in data) {
        let value = data[key]
        if (key === 'status' || key === 'even_odd_input' || key == 'id' || key === 'name') continue
        if (key === 'type') {
            result = `${value}${result}`
        } else {
            value = pyBoolMap[value] ?? value
            result += ` --${key} ${value}`
        }
    }
    return 'isonet.py ' + result
}

export default toCommand
