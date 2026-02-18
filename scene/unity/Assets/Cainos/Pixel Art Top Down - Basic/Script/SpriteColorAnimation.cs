using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cainos.PixelArtTopDown_Basic
{
    // анимируем цвет спрайта по градиенту во времени
    public class SpriteColorAnimation : MonoBehaviour
    {
        public Gradient gradient;
        public float time;

        private SpriteRenderer sr;
        private float timer;

        private void Start()
        {
            timer = time * Random.value;
            sr = GetComponent<SpriteRenderer>();
        }

        private void Update()
        {
            if (sr)
            {
                timer += Time.deltaTime;
                if (timer > time) timer = 0.0f;

                sr.color = gradient.Evaluate(timer / time);
            }
        }
    }
}
