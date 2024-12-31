import json
import pygame as pg
import sys
import random
import p2v

def my_p2v(phonemes, delta=0.0):
    def v(p):
        if p in ['', '-', '¡', ',']: return 'X'
        if p in ['dh', 'th', 's', 'z', 'sh', 'zh']: return 'A'
        if p in ['ah']: return 'C'
        if p in ['f', 'V']: return 'G'
        if p in ['iy']: return 'B'
        if p in ['l']: return 'H'
        if p in ['d', 't']: return 'B' # CHANGED
        if p in ['eh']: return 'C'
        if p in ['k', 'g']: return 'B'
        if p in ['uw']: return 'F'
        if p in ['p', 'b']: return 'A'
        if p in ['ch', 'jh']: return 'F'
        if p in ['hh']: return 'C'
        if p in ['ae']: return 'C'
        if p in ['n']: return 'A'
        if p in ['r']: return 'C' # 'E' # BEF
        if p in ['ih']: return 'B'
        if p in ['m']: return 'A'
        if p in ['ay']: return 'D' # dipthong DB
        if p in ['er']: return 'E' ##
        if p in ['w']: return 'F'
        if p in ['ng']: return 'C' # Changed
        if p in ['aa']: return 'B' # Changed
        if p in ['uh']: return 'F'
        if p in ['aw']: return 'C' # Changed to C
        raise Exception(f'Missing phoneme {p}')
    return [
        {
            'viseme': v(ph['phoneme']),
            'time': ph['time'] + delta,
        }
        for ph in phonemes
    ]

def main():
    data = json.loads(open('test2.json').read())
    pdata = json.loads(open('data.json').read())
    screen = pg.display.set_mode((1024, 768))
    clock = pg.time.Clock()
    talk = pg.mixer.music.load('test2.ogg')
    done = False
    playing = False
    start_time = 0
    state = 'X'
    last_transition_time = 0
    next_state = 'X'
    cue_map = {
        'X': 'lisa-X',
        'A': 'lisa-A',
        'B': 'lisa-B',
        'C': 'lisa-C',
        'D': 'lisa-D',
        'E': 'lisa-E',
        'F': 'lisa-F',
        'G': 'lisa-G',
        'H': 'lisa-H',
    }
    phonemes = pdata['phonemes']
    print(' '.join([ph['phoneme'] for ph in phonemes]))
    imgs = {
        key: pg.image.load(f'{cue_map[key]}.png')
        for key in cue_map
    }
    delta = 0
    # visemes = [ {
    #     'viseme': cue['value'],
    #     'time': cue['start'] * 1000 + delta,
    # } for cue in data['mouthCues']]
    visemes = my_p2v(phonemes, delta=-70)
    original_visemes = visemes[:]

    while not done:
        screen.blit(imgs[state], (0, 0))
        pg.display.flip()
        if playing:
            time = pg.time.get_ticks() - start_time
            while len(visemes) > 0 and visemes[0]['time'] < time:
                viseme = visemes.pop(0)
                sys.stdout.write(f"{viseme['viseme']} ")
                sys.stdout.flush()
                next_state = viseme['viseme']
            if len(visemes) == 0:
                next_state = 'X'
            if time - last_transition_time > 80:
                state = next_state
                last_transition_time = time
            if time - last_transition_time > 100:
                state = 'A'

        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            elif event.type == pg.MOUSEBUTTONDOWN:
                start_time = pg.time.get_ticks()
                playing = True
                pg.mixer.music.play()
                visemes = original_visemes[:]
                state = 'X'
                last_transition_time = 0
                next_state = 'X'
                print('\n')


if __name__ == '__main__':
    pg.init()
    main()
    pg.quit()
